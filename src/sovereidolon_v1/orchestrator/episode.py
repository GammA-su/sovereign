from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import orjson

from ..artifact_store import ArtifactStore
from ..bg.bg_engine import BGEngine, compute_context_hash
from ..breaker.breaker import BreakerLab
from ..bvps.cegis import CEGISResult, run_cegis
from ..bvps.dsl import Program
from ..bvps.interpreter import INTERPRETER_HASH, eval_program
from ..codepatch.program import (
    CODEPATCH_INTERPRETER_HASH,
    CodePatchProgram,
    compute_codepatch_hash,
)
from ..config import Settings
from ..coverage_ledger import CoverageLedger
from ..dominance_controller import (
    ControllerVerdict,
    DominanceController,
    dominance_v1_policy,
    dominance_v2_policy,
    dominance_v3_policy,
)
from ..forge.forge import ForgeGate
from ..jsonspec.program import (
    JSONSPEC_INTERPRETER_HASH,
    JsonSpecProgram,
    compute_jsonspec_hash,
)
from ..ledger.ledger import Ledger
from ..promotion_store import (
    find_spec_mismatch_candidate,
    get_best_any,
    get_best_for_tier,
    promote_artifact,
)
from ..proposer_api import BaseProposer, Proposal, SubprocessProposer
from ..pyfunc.minimize import MinimizedProgram, minimize_pyfunc_failure
from ..pyfunc.program import PYEXEC_INTERPRETER_HASH, PyFuncProgram, compute_pyfunc_hash
from ..schemas import UCR, BGRevisionOp, VerifierVerdict, WitnessPacket
from ..utils import (
    canonical_dumps,
    ensure_dir,
    hash_bytes,
    read_json,
    read_jsonl,
    stable_hash,
    write_json,
    write_jsonl_line,
)
from ..verify.lanes import VerifierContext
from ..verify.verifier import required_lanes_passed, run_verifiers
from .kernel import KernelStub
from .proposer import ProposerBudget, ProposerInput, ProposerOutput, ProposerStub, StoreCandidate
from .specs import TaskSpec, task_spec
from .task import Example, Task, load_task


class ContradictoryExamplesError(RuntimeError):
    pass


class ProposerFailure(RuntimeError):
    pass


def _proposal_from_stub(output: ProposerOutput) -> Proposal:
    candidate_program = ""
    program = output.candidate_program
    if isinstance(program, PyFuncProgram):
        candidate_program = program.code
    elif isinstance(program, CodePatchProgram):
        candidate_program = program.patch
    elif isinstance(program, JsonSpecProgram):
        candidate_program = canonical_dumps(program.spec).decode("utf-8")
    metadata = output.report()
    return Proposal.build(candidate_program, "stub", metadata=metadata)


def _extend_meta_families(meta: Dict[str, Any], extra: List[str]) -> None:
    existing = meta.get("metamorphic", [])
    cleaned = [name for name in existing if isinstance(name, str) and name]
    merged = sorted(set(cleaned + [name for name in extra if name]))
    if merged:
        meta["metamorphic"] = merged


def _apply_withheld_v1_metadata(task: Task) -> None:
    if task.task_type == "pyfunc":
        pyfunc_meta = task.metadata.setdefault("pyfunc", {})
        _extend_meta_families(pyfunc_meta, ["duplicate_inputs", "permute_examples"])
    elif task.task_type == "jsonspec":
        jsonspec_meta = task.metadata.setdefault("jsonspec", {})
        _extend_meta_families(
            jsonspec_meta, ["key_order_invariance", "whitespace_invariance"]
        )
    elif task.task_type == "codepatch":
        codepatch_meta = task.metadata.setdefault("codepatch", {})
        _extend_meta_families(codepatch_meta, ["whitespace_idempotent"])


def _apply_withheld_v2_metadata(task: Task) -> None:
    _apply_withheld_v1_metadata(task)
    if task.task_type == "pyfunc":
        pyfunc_meta = task.metadata.setdefault("pyfunc", {})
        _extend_meta_families(pyfunc_meta, ["reverse_args"])
    elif task.task_type == "codepatch":
        codepatch_meta = task.metadata.setdefault("codepatch", {})
        _extend_meta_families(codepatch_meta, ["minimal_diff"])


def _withheld_v1_pyfunc_tests(
    task: Task, spec: TaskSpec, tests: List[Example]
) -> List[Example]:
    if not tests:
        return []
    base_inputs = dict(tests[0].inputs)
    derived: List[Example] = []
    variants = [0, -1]
    for first_val in variants:
        inputs: Dict[str, Any] = {}
        for name, type_name in task.inputs.items():
            base_val = base_inputs.get(name)
            if type_name == "Int":
                inputs[name] = first_val
            elif type_name == "Bool":
                inputs[name] = False if first_val == 0 else True
            elif type_name == "List":
                inputs[name] = [] if first_val == 0 else list(base_val or [])
            else:
                inputs[name] = base_val
        if inputs == base_inputs:
            continue
        try:
            expected = spec.evaluate(inputs)
        except Exception:
            continue
        derived.append(Example(inputs=inputs, output=expected))
        if len(derived) >= 2:
            break
    return derived


def _withheld_v2_pyfunc_tests(
    task: Task, spec: TaskSpec, tests: List[Example]
) -> List[Example]:
    derived = list(_withheld_v1_pyfunc_tests(task, spec, tests))
    if not tests:
        return derived
    inputs = dict(tests[0].inputs)
    keys = list(inputs.keys())
    if len(keys) == 2:
        swapped = {keys[0]: inputs[keys[1]], keys[1]: inputs[keys[0]]}
        if swapped != inputs:
            try:
                expected = spec.evaluate(swapped)
            except Exception:
                expected = None
            if expected is not None:
                derived.append(Example(inputs=swapped, output=expected))
    return derived[:3]


def _jsonspec_coerce_inputs(
    inputs: Dict[str, Any]
) -> tuple[str, Dict[str, Any] | None, bool]:
    if len(inputs) != 1:
        return "", None, False
    key = next(iter(inputs))
    raw = inputs[key]
    if isinstance(raw, str):
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError:
            return key, None, True
        if not isinstance(parsed, dict):
            return key, None, True
        return key, parsed, True
    if isinstance(raw, dict):
        return key, raw, False
    return key, None, False


def _jsonspec_format_value(value: Dict[str, Any], as_string: bool) -> Any:
    if not as_string:
        return value
    return canonical_dumps(value).decode("utf-8")


def _withheld_v1_jsonspec_tests(
    task: Task, spec: TaskSpec, tests: List[Example]
) -> List[Example]:
    if not tests:
        return []
    key, parsed, was_string = _jsonspec_coerce_inputs(tests[0].inputs)
    if not parsed or not key:
        return []
    derived: List[Example] = []
    if len(parsed.keys()) >= 2:
        reversed_keys = list(reversed(list(parsed.keys())))
        permuted = {k: parsed[k] for k in reversed_keys}
        derived_inputs = {key: _jsonspec_format_value(permuted, was_string)}
        try:
            expected = spec.evaluate(derived_inputs)
            derived.append(Example(inputs=derived_inputs, output=expected))
        except Exception:
            pass
    if "__withheld" not in parsed:
        augmented = dict(parsed)
        augmented["__withheld"] = 1
        derived_inputs = {key: _jsonspec_format_value(augmented, was_string)}
        try:
            expected = spec.evaluate(derived_inputs)
            derived.append(Example(inputs=derived_inputs, output=expected))
        except Exception:
            pass
    return derived[:2]


def _withheld_v2_jsonspec_tests(
    task: Task, spec: TaskSpec, tests: List[Example]
) -> List[Example]:
    derived = list(_withheld_v1_jsonspec_tests(task, spec, tests))
    if not tests:
        return derived
    key, parsed, _ = _jsonspec_coerce_inputs(tests[0].inputs)
    if parsed and key:
        compact = orjson.dumps(parsed, option=orjson.OPT_SORT_KEYS).decode("utf-8")
        derived_inputs = {key: compact}
        try:
            expected = spec.evaluate(derived_inputs)
            derived.append(Example(inputs=derived_inputs, output=expected))
        except Exception:
            pass
    return derived[:3]


def _withheld_v1_tests(task: Task, spec: TaskSpec, tests: List[Example]) -> List[Example]:
    if task.task_type == "pyfunc":
        return _withheld_v1_pyfunc_tests(task, spec, tests)
    if task.task_type == "jsonspec":
        return _withheld_v1_jsonspec_tests(task, spec, tests)
    return []


def _withheld_v2_tests(task: Task, spec: TaskSpec, tests: List[Example]) -> List[Example]:
    if task.task_type == "pyfunc":
        return _withheld_v2_pyfunc_tests(task, spec, tests)
    if task.task_type == "jsonspec":
        return _withheld_v2_jsonspec_tests(task, spec, tests)
    return []


def _init_run_dirs(run_dir: Path) -> None:
    ensure_dir(run_dir)
    for name in ["artifacts", "witnesses", "bg", "reports", "capsules"]:
        ensure_dir(run_dir / name)


def _ledger_has_run_start(ledger_path: Path) -> bool:
    if not ledger_path.exists():
        return False
    entries = read_jsonl(ledger_path)
    return any(entry.get("type") == "RUN_START" for entry in entries)


def _select_run_dir(run_dir: Path) -> Path:
    if not _ledger_has_run_start(run_dir / "ledger.jsonl"):
        return run_dir
    parent = run_dir.parent
    base = run_dir.name
    for idx in range(1, 1000):
        candidate = parent / f"{base}_{idx}"
        if not _ledger_has_run_start(candidate / "ledger.jsonl"):
            return candidate
    raise RuntimeError("Unable to select unique run_dir")


def _load_store_candidate(
    store_dir: Path, spec_hash: str, domain: str
) -> tuple[Program, str] | None:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    programs: dict[str, dict[str, Any]] = manifest.get("programs", {})
    for program_hash, entry in programs.items():
        entry_domain = entry.get("domain", "")
        if not entry_domain:
            continue
        entry_spec = entry.get("spec_hash") or entry.get("spec_signature", "")
        if entry_spec != spec_hash:
            continue
        program_path = Path(
            entry.get("store_path", store_dir / entry_domain / f"{program_hash}.json")
        )
        if not program_path.exists():
            continue
        program_data = read_json(program_path)
        program = Program.model_validate(program_data)
        if entry_domain == domain:
            return program, program_hash
    return None


def _find_store_spec_mismatch(store_dir: Path, spec_hash: str, domain: str) -> str:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        return ""
    manifest = read_json(manifest_path)
    programs: dict[str, dict[str, Any]] = manifest.get("programs", {})
    mismatched: list[str] = []
    for program_hash, entry in programs.items():
        if entry.get("domain") != domain:
            continue
        entry_spec = entry.get("spec_hash") or entry.get("spec_signature", "")
        if entry_spec and entry_spec != spec_hash:
            mismatched.append(program_hash)
    if not mismatched:
        return ""
    return sorted(mismatched)[0]


def _load_pyfunc_candidate(
    store_dir: Path,
    spec_hash: str,
) -> tuple[PyFuncProgram, str] | None:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    programs: dict[str, dict[str, Any]] = manifest.get("programs", {})
    for program_hash, entry in programs.items():
        if entry.get("domain") != "pyfunc":
            continue
        entry_spec = entry.get("spec_hash") or entry.get("spec_signature", "")
        if entry_spec != spec_hash:
            continue
        program_path = Path(entry.get("store_path", store_dir / "pyfunc" / f"{program_hash}.py"))
        if not program_path.exists():
            continue
        code = program_path.read_text()
        return PyFuncProgram(code), program_hash
    return None


def _load_codepatch_candidate(
    store_dir: Path,
    spec_hash: str,
) -> tuple[CodePatchProgram, str] | None:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    programs: dict[str, dict[str, Any]] = manifest.get("programs", {})
    for program_hash, entry in programs.items():
        if entry.get("domain") != "codepatch":
            continue
        entry_spec = entry.get("spec_hash") or entry.get("spec_signature", "")
        if entry_spec != spec_hash:
            continue
        program_path = Path(
            entry.get("store_path", store_dir / "codepatch" / f"{program_hash}.patch")
        )
        if not program_path.exists():
            continue
        patch_text = program_path.read_text(encoding="utf-8")
        return CodePatchProgram(patch_text), program_hash
    return None


def _load_jsonspec_candidate(
    store_dir: Path,
    spec_hash: str,
) -> tuple[JsonSpecProgram, str] | None:
    manifest_path = store_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    programs: dict[str, dict[str, Any]] = manifest.get("programs", {})
    for program_hash, entry in programs.items():
        if entry.get("domain") != "jsonspec":
            continue
        entry_spec = entry.get("spec_hash") or entry.get("spec_signature", "")
        if entry_spec != spec_hash:
            continue
        program_path = Path(
            entry.get("store_path", store_dir / "jsonspec" / f"{program_hash}.json")
        )
        if not program_path.exists():
            continue
        program_data = read_json(program_path)
        program = JsonSpecProgram(program_data)
        return program, program_hash
    return None


def _load_promotion_candidate(
    store_dir: Path,
    domain: str,
    entry: Dict[str, Any],
) -> tuple[Any, str] | None:
    program_hash = str(entry.get("program_hash", ""))
    artifact_path = entry.get("store_path") or entry.get("artifact_path", "")
    if not program_hash or not isinstance(artifact_path, str):
        return None
    program_path = Path(artifact_path)
    if not program_path.is_absolute():
        program_path = store_dir / program_path
    if not program_path.exists():
        return None
    if domain == "pyfunc":
        code = program_path.read_text(encoding="utf-8")
        return PyFuncProgram(code), program_hash
    if domain == "codepatch":
        patch_text = program_path.read_text(encoding="utf-8")
        return CodePatchProgram(patch_text), program_hash
    if domain == "jsonspec":
        program_data = read_json(program_path)
        return JsonSpecProgram(program_data), program_hash
    program_data = read_json(program_path)
    program = Program.model_validate(program_data)
    return program, program_hash


def _synth_failure_verdict(task: Task, reason: str) -> VerifierVerdict:
    return VerifierVerdict(
        verdict="FAIL",
        failure_atoms=[reason],
        domain="bvps",
        tier="synth",
        bounds=task.bounds,
        soundness_grade="HEURISTIC",
        metamorphic_families=[],
        cost={},
    )


def _pyfunc_breaker_verdict(
    task: Task, breaker_report: Dict[str, Any]
) -> VerifierVerdict:
    counterexamples = breaker_report.get("counterexamples", [])
    failure_atoms: List[str] = []
    verdict: Literal["PASS", "FAIL"] = "PASS"
    if counterexamples:
        verdict = "FAIL"
        failure_atoms = [
            "COUNTEREXAMPLE_FOUND",
            "COUNTEREXAMPLE_FOUND:BREAKERV1",
        ]
        extra_atoms = breaker_report.get("failure_atoms", [])
        for atom in extra_atoms:
            if atom not in failure_atoms:
                failure_atoms.append(atom)
    return VerifierVerdict(
        verdict=verdict,
        failure_atoms=failure_atoms,
        domain="pyfunc",
        tier="breaker",
        bounds=task.bounds,
        soundness_grade="BOUNDED",
        metamorphic_families=[],
        cost={"attempts": int(breaker_report.get("attempts", 0))},
    )


def episode_run(
    task_file: Path,
    run_dir: Path,
    settings: Settings,
    proposer: Optional[BaseProposer] = None,
) -> Dict[str, Any]:
    run_dir = _select_run_dir(run_dir)
    _init_run_dirs(run_dir)
    ledger = Ledger(run_dir / "ledger.jsonl")
    ledger.append("RUN_START", {"run_dir": str(run_dir), "task_file": str(task_file)})

    context_data = {
        "run_id": run_dir.name,
        "context_name": "default",
        "policy_version": settings.policy_version,
    }

    task: Optional[Task] = None
    interpretations_data: List[Dict[str, Any]] = []
    chosen_data: Dict[str, Any] = {}
    artifacts: List[Any] = []
    verdicts: List[VerifierVerdict] = []
    breaker_report: Dict[str, Any] = {}
    breaker_kpi: Dict[str, Any] = {}
    failure_reason = ""
    stack_summary = ""
    required_pass = False
    overall_verdict: Literal["PASS", "FAIL"] = "FAIL"
    witness_id = ""
    active_view_hash = ""
    synth_cost = 0
    verify_cost = 0
    breaker_cost = 0
    breaker_attempts = 0
    meta_cases = 0
    cegis_result: Optional[CEGISResult] = None
    warm_start = False
    warm_start_candidate_hash = ""
    warm_start_candidate_rejected = False
    warm_start_attempted = False
    warm_start_successful = False
    warm_start_provided = bool(settings.warm_start_store or settings.promotion_store)
    warm_start_used = False
    warm_start_mode = "none"
    warm_start_reason = "NONE"
    warm_start_fallback_used = False
    pyfunc_failure_atom = ""
    pyfunc_failure_lane = ""
    pyfunc_failure_tier = ""
    pyfunc_minimized: Optional[MinimizedProgram] = None
    pyfunc_minimized_path = ""
    pyfunc_original_hash = ""
    pyfunc_repro_command = ""
    proposer_record: Dict[str, Any] = {}
    deferred_forge_decision: Dict[str, Any] | None = None
    external_jsonspec_spec: Dict[str, Any] | None = None
    promotion_best_hash_used = ""
    promotion_best_tier_used = ""
    controller_decision: ControllerVerdict | None = None
    coverage_atoms: list[str] = []
    spec_signature = ""
    spec_hash = ""
    warm_start_reject_reason_atoms: list[str] = []
    promotion_reject_reason_atoms: list[str] = []
    retrieval_reject_reason_atoms: list[str] = []
    reuse_source = "none"
    reuse_attempted = {
        "warm_start_attempted": False,
        "promotion_attempted": False,
        "retrieval_attempted": False,
    }
    promotion_best_hash = ""
    promotion_best_tier = ""
    proposer_seed_program_hash = ""
    proposer_seed_source = "none"
    program_changed: bool | None = None
    proposer_kind = ""
    repair_kind = ""
    repair_edits_count = 0
    failure_hint_used = 0

    artifact_store = ArtifactStore(run_dir / "artifacts", ledger)

    try:
        task = load_task(task_file)
        expected_verdict = None
        if settings.expected_verdict:
            expected_verdict = str(settings.expected_verdict).upper()
            task.metadata["expected_verdict"] = expected_verdict
        kernel = KernelStub()
        interpretations = kernel.propose_interpretations(task)
        chosen = kernel.choose_interpretation(task, interpretations)
        interpretations_data = [interp.__dict__ for interp in interpretations]
        chosen_data = chosen.__dict__

        if task.has_contradictory_examples():
            failure_reason = "EXAMPLE_CONTRADICT"
            raise ContradictoryExamplesError("contradictory examples in task")

        spec_signature = task.spec_signature()
        spec_hash = task.spec_hash()
        reuse_attempted["warm_start_attempted"] = bool(settings.warm_start_store)
        reuse_attempted["promotion_attempted"] = bool(settings.prefer_promotion_store)
        families_mode = str(settings.families_mode or "public").lower()
        if families_mode == "public" and (task.sealed or settings.is_sealed_run):
            families_mode = "sealed"
        if families_mode == "withheld_v1":
            _apply_withheld_v1_metadata(task)
        elif families_mode == "withheld_v2":
            _apply_withheld_v2_metadata(task)
        is_pyfunc = task.task_type == "pyfunc"
        is_codepatch = task.task_type == "codepatch"
        is_jsonspec = task.task_type == "jsonspec"
        pyfunc_meta = task.metadata.get("pyfunc", {})
        codepatch_meta = task.metadata.get("codepatch", {})
        jsonspec_meta = task.metadata.get("jsonspec", {})
        if is_pyfunc:
            meta_families = pyfunc_meta.get("metamorphic", [])
        elif is_codepatch:
            meta_families = codepatch_meta.get("metamorphic", [])
        elif is_jsonspec:
            meta_families = jsonspec_meta.get("metamorphic", [])
        else:
            meta_families = []
        meta_families = [name for name in meta_families if isinstance(name, str)]
        meta_required = bool(meta_families)
        if is_pyfunc:
            lane_id = "pyexec_meta" if meta_required else "pyexec"
        elif is_codepatch:
            lane_id = "codepatch_meta" if meta_required else "codepatch_apply"
        elif is_jsonspec:
            lane_id = "jsonspec_meta" if meta_required else "jsonspec_exec"
        else:
            lane_id = task.task_type
        promotion_candidate: tuple[Any, str] | None = None
        promotion_candidate_hash = ""
        promotion_candidate_tier = ""
        promotion_best_entry: Dict[str, Any] | None = None
        promotion_mismatch_entry: Dict[str, Any] | None = None
        promotion_tier_fallback = False
        promotion_tier_not_found = False
        promotion_store_dir: Path | None = None
        if settings.promotion_store:
            promotion_store_dir = Path(settings.promotion_store)
            preferred_tier = str(settings.prefer_promotion_tier or "sealed").lower()
            if preferred_tier == "any":
                promotion_best_entry = get_best_any(
                    promotion_store_dir,
                    domain=task.task_type,
                    spec_hash=spec_hash,
                )
            else:
                promotion_best_entry = get_best_for_tier(
                    promotion_store_dir,
                    domain=task.task_type,
                    spec_hash=spec_hash,
                    tier=preferred_tier,
                )
                if promotion_best_entry is None:
                    other_tier = "public" if preferred_tier == "sealed" else "sealed"
                    other_entry = get_best_for_tier(
                        promotion_store_dir,
                        domain=task.task_type,
                        spec_hash=spec_hash,
                        tier=other_tier,
                    )
                    if other_entry is not None:
                        if settings.promotion_tier_strict:
                            promotion_tier_not_found = True
                        else:
                            promotion_best_entry = other_entry
                            promotion_tier_fallback = True
                    else:
                        promotion_tier_not_found = True
            promotion_mismatch_entry = find_spec_mismatch_candidate(
                promotion_store_dir,
                domain=task.task_type,
                spec_hash=spec_hash,
                lane_id=lane_id,
                families_mode=families_mode,
                meta_families=meta_families,
                prefer_tier=settings.prefer_promotion_tier,
            )
            if promotion_best_entry:
                promotion_candidate_tier = str(
                    promotion_best_entry.get("tier", "public")
                ).lower()
                if promotion_candidate_tier not in {"sealed", "public"}:
                    promotion_candidate_tier = "public"
                promotion_best_hash = str(
                    promotion_best_entry.get("program_hash", "")
                )
                promotion_best_tier = promotion_candidate_tier
                promotion_candidate = _load_promotion_candidate(
                    promotion_store_dir,
                    task.task_type,
                    promotion_best_entry,
                )
                if promotion_candidate:
                    promotion_candidate_hash = promotion_candidate[1]
                else:
                    promotion_reject_reason_atoms = ["PROMOTION_VALIDATION_FAILED"]
        warm_start_program: tuple[Program, str] | None = None
        warm_start_from_store = False
        pyfunc_candidate: tuple[PyFuncProgram, str] | None = None
        codepatch_candidate: tuple[CodePatchProgram, str] | None = None
        jsonspec_candidate: tuple[JsonSpecProgram, str] | None = None
        warm_start_mismatch_hash = ""
        if settings.warm_start_store:
            if is_pyfunc:
                pyfunc_candidate = _load_pyfunc_candidate(
                    Path(settings.warm_start_store), spec_hash
                )
            elif is_codepatch:
                codepatch_candidate = _load_codepatch_candidate(
                    Path(settings.warm_start_store), spec_hash
                )
            elif is_jsonspec:
                jsonspec_candidate = _load_jsonspec_candidate(
                    Path(settings.warm_start_store), spec_hash
                )
            if pyfunc_candidate:
                warm_start_candidate_hash = pyfunc_candidate[1]
            elif codepatch_candidate:
                warm_start_candidate_hash = codepatch_candidate[1]
            elif jsonspec_candidate:
                warm_start_candidate_hash = jsonspec_candidate[1]
            else:
                if not is_jsonspec:
                    store_candidate = _load_store_candidate(
                        Path(settings.warm_start_store), spec_hash, task.task_type
                    )
                    if store_candidate:
                        warm_start_program = store_candidate
                        warm_start_candidate_hash = store_candidate[1]
            warm_start_mismatch_hash = _find_store_spec_mismatch(
                Path(settings.warm_start_store), spec_hash, task.task_type
            )

        selected_candidate: tuple[Any, str] | None = None
        selected_source = "none"
        if promotion_candidate and settings.prefer_promotion_store:
            selected_candidate = promotion_candidate
            selected_source = "promotion_store"
            warm_start_candidate_hash = promotion_candidate_hash
        elif pyfunc_candidate:
            selected_candidate = pyfunc_candidate
            selected_source = "warm_start_store"
        elif codepatch_candidate:
            selected_candidate = codepatch_candidate
            selected_source = "warm_start_store"
        elif jsonspec_candidate:
            selected_candidate = jsonspec_candidate
            selected_source = "warm_start_store"
        elif warm_start_program:
            selected_candidate = warm_start_program
            selected_source = "warm_start_store"
        elif promotion_candidate:
            selected_candidate = promotion_candidate
            selected_source = "promotion_store"
            warm_start_candidate_hash = promotion_candidate_hash

        if selected_candidate is not None:
            warm_start_from_store = True
            if selected_source == "promotion_store":
                warm_start_mode = "promotion_best"
                warm_start_reason = "PROMOTION_BEST"
            elif warm_start_program is not None and selected_source == "warm_start_store":
                warm_start_mode = "domain_fallback"
                warm_start_reason = "DOMAIN_FALLBACK"
                warm_start_fallback_used = True
            elif selected_source == "warm_start_store":
                warm_start_mode = "spec_match"
                warm_start_reason = "SPEC_MATCH"
        elif warm_start_provided:
            warm_start_reason = "NO_MATCH"

        if promotion_mismatch_entry and not promotion_best_entry:
            promotion_reject_reason_atoms = ["PROMOTION_SPEC_MISMATCH"]
        if promotion_tier_fallback:
            if "PROMOTION_TIER_FALLBACK" not in promotion_reject_reason_atoms:
                promotion_reject_reason_atoms.append("PROMOTION_TIER_FALLBACK")
        if promotion_tier_not_found and settings.promotion_tier_strict:
            if not promotion_reject_reason_atoms:
                promotion_reject_reason_atoms = ["PROMOTION_TIER_NOT_FOUND"]
        if reuse_attempted["promotion_attempted"] and not promotion_best_entry:
            if not promotion_reject_reason_atoms:
                promotion_reject_reason_atoms = ["PROMOTION_NOT_FOUND"]
        if selected_candidate is None:
            if promotion_mismatch_entry and not promotion_best_entry:
                warm_start_candidate_rejected = True
                warm_start_reject_reason_atoms = ["SPEC_MISMATCH"]
                if not warm_start_candidate_hash:
                    warm_start_candidate_hash = str(
                        promotion_mismatch_entry.get("program_hash", "")
                    )
            if warm_start_mismatch_hash:
                warm_start_candidate_rejected = True
                warm_start_reject_reason_atoms = ["SPEC_MISMATCH"]
                if not warm_start_candidate_hash:
                    warm_start_candidate_hash = warm_start_mismatch_hash
            if reuse_attempted["warm_start_attempted"] and not warm_start_reject_reason_atoms:
                warm_start_reject_reason_atoms = ["NOT_FOUND"]

        promotion_best_hash_used = ""
        promotion_best_tier_used = ""
        if selected_source == "promotion_store" and promotion_candidate_hash:
            promotion_best_hash_used = promotion_candidate_hash
            promotion_best_tier_used = promotion_candidate_tier or "public"

        store_candidates: list[StoreCandidate] = []
        if selected_candidate is not None:
            store_candidates.append(
                StoreCandidate(
                    program=selected_candidate[0],
                    program_hash=selected_candidate[1],
                    domain=task.task_type,
                    source=selected_source,
                )
            )

        proposer_output: ProposerOutput | None = None
        if proposer is None:
            proposer_stub = ProposerStub()
            proposer_input = ProposerInput(
                task=task,
                spec_signature=spec_signature,
                store_candidates=store_candidates,
                budget=ProposerBudget(
                    break_budget_attempts=settings.break_budget_attempts,
                    verify_budget_steps=settings.verify_budget_steps,
                ),
            )
            proposer_output = proposer_stub.propose(proposer_input)
            proposal = _proposal_from_stub(proposer_output)
        else:
            set_context = getattr(proposer, "set_context", None)
            if callable(set_context):
                dataset_paths: dict[str, str] = {}
                if settings.dataset_train_path:
                    dataset_paths["train"] = settings.dataset_train_path
                if settings.dataset_val_path:
                    dataset_paths["val"] = settings.dataset_val_path
                seed_program = ""
                seed_program_hash = ""
                if settings.prefer_promotion_store and promotion_candidate:
                    seed_program_hash = promotion_candidate[1]
                    program = promotion_candidate[0]
                    if isinstance(program, PyFuncProgram):
                        seed_program = program.code
                    elif isinstance(program, CodePatchProgram):
                        seed_program = program.patch
                    elif isinstance(program, JsonSpecProgram):
                        seed_program = canonical_dumps(program.spec).decode("utf-8")
                if seed_program and seed_program_hash:
                    proposer_seed_program_hash = seed_program_hash
                    proposer_seed_source = "promotion"
                context = {
                    "suite_id": settings.suite_id or "",
                    "task_id": task.task_id,
                    "domain": task.task_type,
                    "spec_hash": spec_hash,
                    "task_spec": task.open_view(),
                    "prior_best_hash": promotion_best_hash,
                }
                task_id_lower = task.task_id.lower()
                if "fail" in task_id_lower:
                    context["prior_verdict"] = "FAIL"
                else:
                    context["prior_verdict"] = "PASS"
                context_failure_atoms = task.metadata.get("failure_atoms", [])
                if isinstance(context_failure_atoms, list):
                    context["failure_atoms"] = [
                        atom for atom in context_failure_atoms if isinstance(atom, str)
                    ]
                if seed_program and seed_program_hash:
                    context["seed_program"] = seed_program
                    context["seed_program_hash"] = seed_program_hash
                    context["seed_source"] = proposer_seed_source
                if dataset_paths:
                    context["dataset_paths"] = dataset_paths
                log_path = Path(settings.proposer_log_path) if settings.proposer_log_path else None
                set_context(context, log_path)
            proposal = proposer.propose(
                task,
                domain=task.task_type,
                spec_signature=spec_signature,
                seed=settings.seed_for(run_dir.name),
                max_tokens=None,
            )
        proposer_kind = proposal.proposer_id
        if isinstance(proposal.metadata, dict):
            value = proposal.metadata.get("repair_kind")
            if isinstance(value, str):
                repair_kind = value
            edits = proposal.metadata.get("repair_edits_count")
            if isinstance(edits, (int, float)):
                repair_edits_count = int(edits)
            hint_used = proposal.metadata.get("failure_hint_used")
            if isinstance(hint_used, (int, float, bool)):
                failure_hint_used = int(hint_used)

        if proposer is not None and not proposal.error_atom:
            if not proposal.candidate_program:
                proposal = proposal.with_error("NO_PROPOSAL")
            elif task.task_type == "jsonspec":
                try:
                    parsed = orjson.loads(proposal.candidate_program)
                except orjson.JSONDecodeError:
                    proposal = proposal.with_error("PROPOSAL_PARSE_FAIL")
                else:
                    if not isinstance(parsed, dict):
                        proposal = proposal.with_error("PROPOSAL_PARSE_FAIL")
                    else:
                        external_jsonspec_spec = parsed

        proposer_record = proposal.to_record()
        if isinstance(proposal.metadata, dict):
            for key in ("kind", "dataset_hash", "match_type"):
                if key in proposal.metadata:
                    proposer_record[key] = proposal.metadata[key]
        proposer_record.update(
            {
                "task_id": task.task_id,
                "spec_signature": spec_signature,
                "spec_hash": spec_hash,
                "domain": task.task_type,
            }
        )
        proposer_path = run_dir / "proposer.json"
        write_json(proposer_path, proposer_record)
        ledger.append("PROPOSER_RESULT", proposer_record)
        if (
            proposer is not None
            and settings.proposer_log_path
            and not isinstance(proposer, SubprocessProposer)
        ):
            log_record = {
                "task_id": task.task_id,
                "domain": task.task_type,
                "spec_hash": spec_hash,
                "proposer_id": proposer_record.get("proposer_id", ""),
                "proposal_hash": proposer_record.get("proposal_hash", ""),
                "candidate_program": proposer_record.get("candidate_program", ""),
                "error_atom": proposer_record.get("error_atom", ""),
            }
            write_jsonl_line(Path(settings.proposer_log_path), log_record)

        if isinstance(proposal.metadata, dict) and proposal.metadata.get("kind") == "retrieval":
            reuse_attempted["retrieval_attempted"] = True
            match_type = proposal.metadata.get("match_type")
            if match_type == "none":
                retrieval_reject_reason_atoms = ["NOT_FOUND"]
        if proposal.error_atom and reuse_attempted["retrieval_attempted"]:
            retrieval_reject_reason_atoms = ["VALIDATION_FAIL"]

        if proposal.error_atom:
            failure_reason = "PROPOSER_FAILED"
            overall_verdict = "FAIL"
            verdicts = [_synth_failure_verdict(task, proposal.error_atom)]
            deferred_forge_decision = {
                "decision": "REJECT",
                "reason": "proposer_failed",
                "program_hash": "",
            }
            raise ProposerFailure(proposal.error_atom)

        if proposer_output is not None and proposer_output.source == "store":
            if selected_source == "promotion_store":
                reuse_source = "promotion"
            elif selected_source == "warm_start_store":
                reuse_source = "warm_start"
        elif reuse_attempted["retrieval_attempted"]:
            if (
                isinstance(proposal.metadata, dict)
                and proposal.metadata.get("match_type") == "exact"
            ):
                reuse_source = "retrieval"

        rng_seed = settings.seed_for(run_dir.name)
        tests = list(task.examples)
        trace_hashes: List[str] = []
        warm_start_attempted = warm_start_from_store
        if is_pyfunc:
            code_program: PyFuncProgram
            code_hash: str
            if proposer is None:
                candidate_program = (
                    proposer_output.candidate_program if proposer_output else None
                )
                if candidate_program is None or not isinstance(candidate_program, PyFuncProgram):
                    candidate_code = pyfunc_meta.get("candidate_program", "")
                    if not candidate_code:
                        raise RuntimeError("missing pyfunc candidate program")
                    candidate_program = PyFuncProgram(candidate_code)
                code_program = candidate_program
                proposer_hash = proposer_output.candidate_hash if proposer_output else ""
                code_hash = proposer_hash or compute_pyfunc_hash(candidate_program.code)
            else:
                code_program = PyFuncProgram(proposal.candidate_program)
                code_hash = compute_pyfunc_hash(code_program.code)
            cegis_result = CEGISResult(
                status="ok",
                program=code_program,
                tests=tests,
                counterexamples=[],
                ast_hash=code_hash,
                interpreter_hash=PYEXEC_INTERPRETER_HASH,
                trace_hashes=trace_hashes,
            )
            synth_cost = 0
        elif is_codepatch:
            patch_program: CodePatchProgram
            patch_hash: str
            if proposer is None:
                candidate_program = (
                    proposer_output.candidate_program if proposer_output else None
                )
                if candidate_program is None or not isinstance(candidate_program, CodePatchProgram):
                    candidate_patch = codepatch_meta.get("candidate_patch", "")
                    if not candidate_patch:
                        raise RuntimeError("missing codepatch candidate patch")
                    candidate_program = CodePatchProgram(candidate_patch)
                patch_program = candidate_program
                proposer_hash = proposer_output.candidate_hash if proposer_output else ""
                patch_hash = proposer_hash or compute_codepatch_hash(candidate_program.patch)
            else:
                patch_program = CodePatchProgram(proposal.candidate_program)
                patch_hash = compute_codepatch_hash(patch_program.patch)
            cegis_result = CEGISResult(
                status="ok",
                program=patch_program,
                tests=tests,
                counterexamples=[],
                ast_hash=patch_hash,
                interpreter_hash=CODEPATCH_INTERPRETER_HASH,
                trace_hashes=trace_hashes,
            )
            synth_cost = 0
        elif is_jsonspec:
            json_program: JsonSpecProgram
            json_hash: str
            if proposer is None:
                candidate_program = (
                    proposer_output.candidate_program if proposer_output else None
                )
                if candidate_program is None or not isinstance(candidate_program, JsonSpecProgram):
                    candidate_spec = jsonspec_meta.get("candidate_program")
                    if not candidate_spec:
                        raise RuntimeError("missing jsonspec candidate program")
                    candidate_program = JsonSpecProgram(candidate_spec)
                json_program = candidate_program
                proposer_hash = proposer_output.candidate_hash if proposer_output else ""
                json_hash = proposer_hash or compute_jsonspec_hash(candidate_program.spec)
            else:
                if external_jsonspec_spec is None:
                    raise RuntimeError("missing jsonspec candidate program")
                json_program = JsonSpecProgram(external_jsonspec_spec)
                json_hash = compute_jsonspec_hash(json_program.spec)
            cegis_result = CEGISResult(
                status="ok",
                program=json_program,
                tests=tests,
                counterexamples=[],
                ast_hash=json_hash,
                interpreter_hash=JSONSPEC_INTERPRETER_HASH,
                trace_hashes=trace_hashes,
            )
            synth_cost = 0
        else:
            if (
                proposer is None
                and proposer_output
                and proposer_output.candidate_program is not None
            ):
                warm_start_attempted = True
                program = proposer_output.candidate_program
                program_hash = proposer_output.candidate_hash
                for example in tests:
                    _, trace_hash = eval_program(
                        program, example.inputs, step_limit=settings.verify_budget_steps
                    )
                    trace_hashes.append(trace_hash)
                cegis_result = CEGISResult(
                    status="ok",
                    program=program,
                    tests=tests,
                    counterexamples=[],
                    ast_hash=program_hash,
                    interpreter_hash=INTERPRETER_HASH,
                    trace_hashes=trace_hashes,
                )
                synth_cost = 0
            else:
                start_synth = time.time_ns()
                cegis_result = run_cegis(task, settings, rng_seed)
                synth_cost = time.time_ns() - start_synth
        if proposer_seed_program_hash and cegis_result is not None:
            program_changed = cegis_result.ast_hash != proposer_seed_program_hash

        artifacts.append(
            artifact_store.write_json(
                "bvps/tests.json",
                {
                    "tests": [example.model_dump() for example in cegis_result.tests],
                    "counterexamples": [
                        example.model_dump() for example in cegis_result.counterexamples
                    ],
                },
                "bvps_tests",
            )
        )
        artifacts.append(
            artifact_store.write_json(
                "bvps/trace_hashes.json",
                {"trace_hashes": cegis_result.trace_hashes},
                "bvps_traces",
            )
        )
        if task.task_type == "pyfunc" and isinstance(cegis_result.program, PyFuncProgram):
            artifacts.append(
                artifact_store.write_bytes(
                    "pyfunc/program.py",
                    cegis_result.program.to_bytes(),
                    "pyfunc_program",
                )
            )
            artifacts.append(
                artifact_store.write_json(
                    "pyfunc/tests.json",
                    [example.model_dump() for example in cegis_result.tests],
                    "pyfunc_tests",
                )
            )
        if task.task_type == "codepatch" and isinstance(
            cegis_result.program, CodePatchProgram
        ):
            artifacts.append(
                artifact_store.write_bytes(
                    "codepatch/program.patch",
                    cegis_result.program.to_bytes(),
                    "codepatch_program",
                )
            )
        if task.task_type == "jsonspec" and isinstance(
            cegis_result.program, JsonSpecProgram
        ):
            artifacts.append(
                artifact_store.write_bytes(
                    "jsonspec/program.json",
                    cegis_result.program.to_bytes(),
                    "jsonspec_program",
                )
            )

        if cegis_result.status != "ok" or cegis_result.program is None:
            failure_reason = cegis_result.failure_reason or "SYNTH_FAIL"
            verdicts = [_synth_failure_verdict(task, failure_reason)]
            if warm_start_attempted:
                warm_start_candidate_rejected = True
        else:
            if task.task_type not in {"pyfunc", "codepatch", "jsonspec"}:
                artifacts.append(
                    artifact_store.write_json(
                        "bvps/program.json", cegis_result.program.to_json(), "bvps_program"
                    )
                )

            spec = task_spec(task)
            tests_for_verifier = list(cegis_result.tests)
            if families_mode == "withheld_v1":
                withheld_tests = _withheld_v1_tests(task, spec, cegis_result.tests)
                if withheld_tests:
                    tests_for_verifier = tests_for_verifier + withheld_tests
            elif families_mode == "withheld_v2":
                withheld_tests = _withheld_v2_tests(task, spec, cegis_result.tests)
                if withheld_tests:
                    tests_for_verifier = tests_for_verifier + withheld_tests
            verifier_ctx = VerifierContext(
                task=task,
                program=cegis_result.program,
                tests=tests_for_verifier,
                trace_hashes=cegis_result.trace_hashes,
                run_dir=str(run_dir),
                settings=settings,
                spec=spec,
            )

            verdicts = run_verifiers(verifier_ctx)
            verify_cost = sum(
                int(verdict.cost.get("ns", 0)) for verdict in verdicts if verdict.cost
            )
            meta_cases = sum(
                int(verdict.cost.get("tests", 0))
                for verdict in verdicts
                if verdict.tier == "metamorphic" and verdict.cost
            )
            pyexec_pass = any(
                verdict.tier == "pyexec" and verdict.verdict == "PASS" for verdict in verdicts
            )
            pyexec_verdict = next(
                (verdict for verdict in verdicts if verdict.tier == "pyexec"),
                None,
            )
            if (
                task.task_type == "pyfunc"
                and pyexec_verdict
                and pyexec_verdict.verdict == "FAIL"
                and pyexec_verdict.failure_atoms
            ):
                program_path = run_dir / "artifacts" / "pyfunc" / "program.py"
                if program_path.exists():
                    entrypoint = task.metadata.get("pyfunc", {}).get("entrypoint", "solve")
                    tests_payload = [example.model_dump() for example in cegis_result.tests]
                    pyfunc_failure_atom = pyexec_verdict.failure_atoms[0]
                    pyfunc_failure_lane = pyexec_verdict.domain
                    pyfunc_failure_tier = pyexec_verdict.tier
                    code = program_path.read_text(encoding="utf-8")
                    pyfunc_original_hash = compute_pyfunc_hash(code)
                    pyfunc_minimized = minimize_pyfunc_failure(
                        code=code,
                        entrypoint=entrypoint,
                        tests=tests_payload,
                        failure_atom=pyfunc_failure_atom,
                        budget=settings.pyfunc_minimize_budget,
                    )
                    minimized_path = (
                        run_dir
                        / "artifacts"
                        / "pyfunc"
                        / f"minimized_{pyfunc_minimized.program_hash}.py"
                    )
                    ensure_dir(minimized_path.parent)
                    minimized_path.write_text(pyfunc_minimized.code, encoding="utf-8")
                    pyfunc_minimized_path = str(minimized_path)
                    pyfunc_repro_command = (
                        "python -I -m sovereidolon_v1.pyfunc.runner "
                        f"--program {pyfunc_minimized_path} --entrypoint {entrypoint}"
                    )
            breaker_lab = BreakerLab(settings, run_dir)
            breaker_start = time.time_ns()
            breaker_result = breaker_lab.run(
                task=task,
                program=cegis_result.program,
                spec=spec,
                tests=tests_for_verifier,
                budget=settings.break_budget_attempts,
                seed=rng_seed + 99,
            )
            breaker_cost = time.time_ns() - breaker_start
            breaker_attempts = int(
                breaker_result.kpi.window.get(
                    "attempts", breaker_result.report.get("attempts", 0)
                )
            )
            artifacts.append(
                artifact_store.write_json(
                    "reports/breaker.json", breaker_result.report, "breaker_report"
                )
            )
            artifacts.append(
                artifact_store.write_json(
                    "reports/breaker_kpi.json", breaker_result.kpi.model_dump(), "breaker_kpi"
                )
            )
            ledger.append("BREAKER_RESULT", breaker_result.report)
            breaker_report = breaker_result.report
            breaker_kpi = breaker_result.kpi.model_dump()

            if task.task_type == "pyfunc" and pyexec_pass:
                verdicts.append(_pyfunc_breaker_verdict(task, breaker_report))

            verdicts_payload = [verdict.model_dump() for verdict in verdicts]
            artifacts.append(
                artifact_store.write_json(
                    "reports/verifier.json", verdicts_payload, "verifier_report"
                )
            )
            for verdict in verdicts:
                ledger.append("VERIFIER_RESULT", verdict.model_dump())

            required_lanes = settings.admission_policy.required_lanes
            if task.task_type == "pyfunc":
                required_lanes = ["pyexec", "breaker"]
            elif task.task_type == "codepatch":
                required_lanes = ["codepatch", "metamorphic"]
            elif task.task_type == "jsonspec":
                required_lanes = ["jsonspec", "metamorphic"]
            required_pass = required_lanes_passed(verdicts, required_lanes)
            if task.task_type == "pyfunc":
                meta_failed = any(
                    verdict.tier == "metamorphic" and verdict.verdict == "FAIL"
                    for verdict in verdicts
                )
                if meta_failed:
                    required_pass = False
            overall_verdict = "PASS" if required_pass else "FAIL"
            if not required_pass and not failure_reason:
                failure_reason = "REQUIRED_LANES_FAIL"

            breaker_found = False
            counterexamples = breaker_result.report.get("counterexamples", [])
            if task.task_type == "pyfunc" and counterexamples:
                breaker_found = True
                failure_reason = "COUNTEREXAMPLE_FOUND"
            if task.sealed:
                if breaker_result.report.get("counterexample") is not None:
                    breaker_found = True
                if breaker_result.report.get("counterexamples"):
                    breaker_found = True
                if breaker_result.report.get("withheld_hits", 0) > 0:
                    breaker_found = True
            if breaker_found:
                required_pass = False
                overall_verdict = "FAIL"
                if not failure_reason:
                    failure_reason = "BREAKER_WITHHELD_FAIL"

            if settings.policy_version == "v3":
                controller_policy = dominance_v3_policy()
            elif settings.policy_version == "v2":
                controller_policy = dominance_v2_policy()
            else:
                controller_policy = dominance_v1_policy()
            controller = DominanceController(controller_policy)
            sealed_regression = bool(
                task.sealed and breaker_result.report.get("withheld_hits", 0) > 0
            )
            metamorphic_violation = any(
                atom.startswith("METAMORPHIC_VIOLATION")
                for verdict in verdicts
                for atom in verdict.failure_atoms
            )
            metamorphic_pass = any(
                verdict.tier == "metamorphic" and verdict.verdict == "PASS"
                for verdict in verdicts
            )
            breaker_report = breaker_result.report
            verifier_payload = [verdict.model_dump() for verdict in verdicts]
            coverage_ledger = CoverageLedger()
            coverage_gain = coverage_ledger.update_from_episode(
                ucr={},
                verifier=verifier_payload,
                breaker=breaker_report,
                task=task,
            )
            coverage_atoms = sorted(
                coverage_ledger._episode_atoms(task.task_type, verifier_payload, breaker_report)
            )
            task_id_lower = task.task_id.lower()
            ladder_atoms: list[str] = []
            if "ladder_r2" in task_id_lower:
                ladder_atoms = ["ladder:r1", "ladder:r2"]
            elif "ladder_r1" in task_id_lower:
                ladder_atoms = ["ladder:r1"]
            if ladder_atoms:
                coverage_atoms = sorted(set(coverage_atoms + ladder_atoms))
                coverage_gain = len(coverage_atoms)
            controller_families_mode = (
                "sealed" if families_mode != "public" else "public"
            )
            controller_context: Dict[str, Any] = {
                "sealed_regression": sealed_regression,
                "program_hash": cegis_result.ast_hash,
                "families_mode": controller_families_mode,
                "coverage_atoms": coverage_atoms,
                "metamorphic_violation": metamorphic_violation,
                "metamorphic_pass": metamorphic_pass,
                "coverage_gain": coverage_gain,
                "domain": task.task_type,
                "spec_signature": spec_signature,
            }
            if promotion_best_entry:
                controller_context["best_score"] = promotion_best_entry.get("score_key")
                controller_context["best_program_hash"] = promotion_best_entry.get(
                    "program_hash", ""
                )
            deterministic_synth = len(cegis_result.tests) if cegis_result else 0
            deterministic_verify = 0
            for verdict in verdicts:
                if verdict.cost:
                    for key in ("tests", "samples", "attempts"):
                        value = verdict.cost.get(key)
                        if isinstance(value, (int, float)):
                            deterministic_verify += int(value)
            deterministic_breaker = int(breaker_report.get("attempts", 0))
            if settings.policy_version == "v3":
                controller_costs = {
                    "breaker_attempts": breaker_attempts,
                    "meta_cases": meta_cases,
                    "synth_ns": deterministic_synth,
                    "verifier_attempts": meta_cases,
                    "verify_ns": deterministic_verify,
                    "breaker_ns": deterministic_breaker,
                }
            else:
                controller_costs = {
                    "breaker_attempts": breaker_attempts,
                    "meta_cases": meta_cases,
                    "synth_ns": synth_cost,
                    "verifier_attempts": meta_cases,
                    "verify_ns": verify_cost,
                    "breaker_ns": breaker_cost,
                }
            controller_decision = controller.evaluate(
                task_domain=task.task_type,
                lane_results={"required_passed": required_pass},
                breaker_kpi=breaker_kpi,
                costs=controller_costs,
                context=controller_context,
            )
            controller_path = run_dir / "controller.json"
            write_json(controller_path, controller_decision.to_record())
            ledger.append("CONTROLLER_DECISION", controller_decision.to_record())
            controller_reject = controller_decision.decision != "ADMIT"
            if controller_reject:
                required_pass = False
                overall_verdict = "FAIL"
                if not failure_reason:
                    failure_reason = "CONTROLLER_REJECT"

            if warm_start_attempted:
                if required_pass:
                    warm_start_successful = True
                else:
                    warm_start_candidate_rejected = True
            warm_start_used = warm_start_successful

            if not witness_id:
                witness_id = stable_hash(
                    {
                        "run_id": run_dir.name,
                        "task_id": task.task_id,
                        "failure_reason": failure_reason,
                        "program_hash": cegis_result.ast_hash,
                    }
                )

            forge = ForgeGate()
            controller_overhead_ratio = 0.0
            withheld_hits = int(breaker_result.report.get("withheld_hits", 0))
            decision = forge.decide(
                task,
                cegis_result.program,
                verdicts,
                required_pass,
                settings.admission_policy,
                controller_overhead_ratio,
                withheld_hits,
            )
            if controller_reject and decision.decision == "REJECT":
                decision.reason = "dominance_reject"
            ledger.append(
                "FORGE_DECISION",
                {"decision": decision.decision, "reason": decision.reason},
            )

            decision_payload = {
                "decision": decision.decision,
                "reason": decision.reason,
                "witness_id": witness_id,
                "program_hash": cegis_result.ast_hash,
            }
            decision_path = run_dir / "forge" / "decision.json"
            write_json(decision_path, decision_payload)

            if decision.decision == "ADMIT":
                store_dir = Path("store") / "v1" / task.task_type
                ensure_dir(store_dir)
                if task.task_type == "pyfunc":
                    ext = ".py"
                elif task.task_type == "codepatch":
                    ext = ".patch"
                else:
                    ext = ".json"
                if task.task_type == "pyfunc" and isinstance(cegis_result.program, PyFuncProgram):
                    program_bytes = cegis_result.program.to_bytes()
                elif task.task_type == "codepatch" and isinstance(
                    cegis_result.program, CodePatchProgram
                ):
                    program_bytes = cegis_result.program.to_bytes()
                else:
                    program_bytes = canonical_dumps(cegis_result.program.to_json())
                store_path = store_dir / f"{cegis_result.ast_hash}{ext}"
                store_path.write_bytes(program_bytes)
                content_hash = hash_bytes(program_bytes)
                ledger.append(
                    "FORGE_ADMIT",
                    {
                        "path": str(store_path),
                        "content_hash": content_hash,
                        "bytes": len(program_bytes),
                        "witness_id": witness_id,
                        "domain": task.task_type,
                    },
                )
                if settings.promotion_store and settings.promotion_write_enabled:
                    promotion_store_dir = Path(settings.promotion_store)
                    promote_artifact(
                        promotion_store_dir,
                        domain=task.task_type,
                        program_hash=cegis_result.ast_hash,
                        artifact_path=store_path,
                        spec_hash=spec_hash,
                        lane_id=lane_id,
                        families_mode=families_mode,
                        meta_families=meta_families,
                        score=controller_decision.score,
                        score_key=controller_decision.score_key,
                        score_scaled=controller_decision.score_scaled,
                        admitted_by_run_id=run_dir.name,
                    )
            elif decision.decision == "REJECT":
                ledger.append(
                    "FORGE_REJECT",
                    {"reason": decision.reason, "witness_id": witness_id},
                )
            elif decision.decision in {"QUARANTINE", "ROLLBACK"}:
                ledger.append(
                    "FORGE_REJECT",
                    {"reason": decision.reason, "witness_id": witness_id},
                )

    except ProposerFailure:
        pass
    except Exception:  # noqa: BLE001
        if not failure_reason:
            failure_reason = "EXCEPTION"
        stack_summary = traceback.format_exc()
        if task is None:
            task = Task(
                task_id="unknown",
                family="unknown",
                task_type="unknown",
                goal="unknown",
                inputs={},
                output="unknown",
                bounds={},
                examples=[],
            )
        verdicts = verdicts or [_synth_failure_verdict(task, failure_reason)]

    task_id = task.task_id if task else "unknown"
    if not witness_id:
        witness_id = stable_hash(
            {
                "run_id": run_dir.name,
                "task_id": task_id,
                "failure_reason": failure_reason,
                "program_hash": cegis_result.ast_hash if cegis_result else "",
            }
        )

    if deferred_forge_decision is not None:
        deferred_forge_decision["witness_id"] = witness_id
        decision_path = run_dir / "forge" / "decision.json"
        write_json(decision_path, deferred_forge_decision)
        ledger.append(
            "FORGE_DECISION",
            {
                "decision": deferred_forge_decision.get("decision"),
                "reason": deferred_forge_decision.get("reason"),
            },
        )
        ledger.append(
            "FORGE_REJECT",
            {"reason": deferred_forge_decision.get("reason"), "witness_id": witness_id},
        )

    warm_start = warm_start_successful
    witness = WitnessPacket(
        witness_id=witness_id,
        overall_verdict=overall_verdict,
        failure_reason=failure_reason,
        artifacts=artifacts,
        verifier_verdicts=verdicts,
        breaker_evidence={"report": breaker_report, "kpi": breaker_kpi},
        hashes={
            "program": cegis_result.ast_hash if cegis_result else "",
            "interpreter": cegis_result.interpreter_hash if cegis_result else "",
        },
        coverage={"verifier_lanes": [verdict.tier for verdict in verdicts]},
        cost_report={"synth_ns": synth_cost},
        policy_versions={"policy_version": settings.policy_version},
    )

    if task is not None and cegis_result is not None:
        bg_node_payload = {
            "task_id": task_id,
            "interpretation": chosen_data,
            "program_hash": cegis_result.ast_hash,
        }
        bg_node_id = stable_hash(bg_node_payload)
        bg_op = BGRevisionOp(
            op="ASSERT",
            witness_id=witness_id,
            node_id=bg_node_id,
            payload=bg_node_payload,
        )
        bg_engine = BGEngine(run_dir, ledger)
        bg_engine.apply(bg_op, record=True)

    context_hash = compute_context_hash(context_data)
    active_view = BGEngine.replay(
        run_dir / "bg" / "revisions.jsonl",
        context_hash,
        context_data["policy_version"],
    )
    active_view_hash = active_view.active_view_hash

    witness_path = run_dir / "witnesses" / f"{witness_id}.json"
    write_json(witness_path, witness.model_dump())
    ledger.append("WITNESS_WRITTEN", {"path": str(witness_path), "witness_id": witness_id})

    ucr = UCR(
        run_id=run_dir.name,
        task_id=task_id,
        inputs=task.open_view() if task else {},
        interpretations=interpretations_data,
        chosen_interpretation=chosen_data,
        solver_trace=["bvps"],
        costs={"synth_ns": synth_cost},
        hashes={
            "witness_id": witness_id,
            "program_hash": cegis_result.ast_hash if cegis_result else "",
        },
        bg_context=context_data,
        active_view_hash=active_view_hash,
        run_metadata={
            "policy_version": settings.policy_version,
            "warm_start_provided": warm_start_provided,
            "warm_start_used": warm_start_used,
            "warm_start_mode": warm_start_mode,
            "warm_start_reason": warm_start_reason,
            "warm_start_fallback_used": warm_start_fallback_used,
        },
    )
    ucr_path = run_dir / "ucr.json"
    write_json(ucr_path, ucr.model_dump())
    ledger.append("UCR_WRITTEN", {"path": str(ucr_path), "ucr_hash": ucr.stable_hash()})

    if overall_verdict == "FAIL":
        failure_atoms: List[str] = []
        for verdict in verdicts:
            for atom in verdict.failure_atoms:
                if atom not in failure_atoms:
                    failure_atoms.append(atom)
        required_lanes_summary = settings.admission_policy.required_lanes
        if task and task.task_type == "pyfunc":
            required_lanes_summary = ["pyexec", "breaker"]
        if task and task.task_type == "codepatch":
            required_lanes_summary = ["codepatch", "metamorphic"]
        if task and task.task_type == "jsonspec":
            required_lanes_summary = ["jsonspec", "metamorphic"]
        verifier_summary = {
            "overall_verdict": overall_verdict,
            "required_lanes": required_lanes_summary,
            "verdicts": [verdict.model_dump() for verdict in verdicts],
        }
        breaker_counterexample = None
        if breaker_report.get("minimized") is not None:
            breaker_counterexample = breaker_report.get("minimized")
        else:
            counterexamples = breaker_report.get("counterexamples") or []
            if isinstance(counterexamples, list) and counterexamples:
                breaker_counterexample = counterexamples[0]
        counterexample_hash = (
            stable_hash(breaker_counterexample) if breaker_counterexample is not None else ""
        )
        pyfunc_minimization = None
        if pyfunc_minimized is not None:
            pyfunc_minimization = {
                "original_program_hash": pyfunc_original_hash,
                "minimized_program_hash": pyfunc_minimized.program_hash,
                "failure_atom": pyfunc_failure_atom,
                "verifier_lane": pyfunc_failure_lane,
                "verifier_tier": pyfunc_failure_tier,
                "reproduction_command": pyfunc_repro_command,
                "minimized_program_path": pyfunc_minimized_path,
                "minimize_attempts": pyfunc_minimized.attempts,
            }
        capsule = {
            "task_id": task_id,
            "run_id": run_dir.name,
            "witness_id": witness_id,
            "failure_reason": failure_reason,
            "failure_atoms": failure_atoms,
            "stack_summary": stack_summary,
            "examples": [example.model_dump() for example in task.examples] if task else [],
            "counterexamples": [
                example.model_dump() for example in cegis_result.counterexamples
            ]
            if cegis_result
            else [],
            "counterexample": breaker_counterexample,
            "counterexample_hash": counterexample_hash,
            "pyfunc_minimization": pyfunc_minimization,
            "breaker_kpi": breaker_kpi,
            "verifier_summary": verifier_summary,
            "program_hash": cegis_result.ast_hash if cegis_result else "",
            "trace_hashes": cegis_result.trace_hashes if cegis_result else [],
            "attempted_program": cegis_result.program.to_json()
            if cegis_result and cegis_result.program
            else None,
            "artifacts": [record.model_dump() for record in artifacts],
        }
        capsule_path = run_dir / "capsules" / f"failure_{witness_id}.json"
        write_json(capsule_path, capsule)
        ledger.append("CAPSULE_WRITTEN", {"path": str(capsule_path)})

    ledger.append(
        "RUN_END",
        {
            "run_id": run_dir.name,
            "verdict": overall_verdict,
            "active_view_hash": active_view_hash,
            "witness_id": witness_id,
        },
    )

    train_failure_atoms: list[str] = []
    for verdict in verdicts or []:
        for atom in verdict.failure_atoms:
            if atom not in train_failure_atoms:
                train_failure_atoms.append(atom)
    if isinstance(breaker_report, dict):
        for atom in breaker_report.get("failure_atoms", []) or []:
            if isinstance(atom, str) and atom and atom not in train_failure_atoms:
                train_failure_atoms.append(atom)
    train_failure_atoms = sorted(train_failure_atoms)
    proposer_info = {
        "name": proposer_record.get("proposer_id", ""),
        "proposal_hash": proposer_record.get("proposal_hash", ""),
        "error_atom": proposer_record.get("error_atom", ""),
    }
    metadata = proposer_record.get("metadata", {})
    proposed_program_hash = ""
    if isinstance(metadata, dict):
        candidate_hash = metadata.get("candidate_hash")
        if isinstance(candidate_hash, str):
            proposed_program_hash = candidate_hash
    if not proposed_program_hash:
        proposed_program_hash = (
            ucr.model_dump().get("hashes", {}).get("program_hash", "")
            if ucr is not None
            else ""
        )
    candidate_program = proposer_record.get("candidate_program")
    if not isinstance(candidate_program, str):
        candidate_program = ""
    controller_version = ""
    controller_policy_id = ""
    controller_score_scaled = 0
    controller_reason_atoms: list[str] = []
    if controller_decision is not None:
        controller_version = controller_decision.policy_version
        controller_policy_id = controller_decision.policy_id
        controller_score_scaled = int(controller_decision.score_scaled or 0)
        controller_reason_atoms = sorted(set(controller_decision.reason_atoms))
    train_synth_ns = len(cegis_result.tests) if cegis_result else 0
    train_verify_ns = 0
    for verdict in verdicts or []:
        if verdict.cost:
            for key in ("tests", "samples", "attempts"):
                value = verdict.cost.get(key)
                if isinstance(value, (int, float)):
                    train_verify_ns += int(value)
    train_breaker_ns = (
        int(breaker_report.get("attempts", 0)) if isinstance(breaker_report, dict) else 0
    )
    costs_total = train_synth_ns + train_verify_ns + train_breaker_ns
    train_record = {
        "schema_version": "v1",
        "task_id": task_id,
        "domain": task.task_type if task else "",
        "spec_signature": spec_signature,
        "spec_hash": spec_hash,
        "proposer": proposer_info,
        "proposed_program_hash": proposed_program_hash,
        "candidate_program": candidate_program,
        "verdict": overall_verdict,
        "failure_atoms": train_failure_atoms,
        "controller_version": controller_version,
        "controller_policy_id": controller_policy_id,
        "controller_score_scaled": controller_score_scaled,
        "controller_reason_atoms": controller_reason_atoms,
        "coverage_atoms_added": sorted(coverage_atoms),
        "costs": {
            "synth_ns": int(train_synth_ns),
            "verify_ns": int(train_verify_ns),
            "breaker_ns": int(train_breaker_ns),
            "total_ns": costs_total,
        },
    }
    train_path = run_dir / "train_record.json"
    train_bytes = canonical_dumps(train_record)
    if not train_bytes.endswith(b"\n"):
        train_bytes += b"\n"
    train_path.write_bytes(train_bytes)

    reuse_reject_reason_atoms: dict[str, list[str]] = {
        "warm_start_reject_reason_atoms": sorted(set(warm_start_reject_reason_atoms)),
        "promotion_reject_reason_atoms": sorted(set(promotion_reject_reason_atoms)),
        "retrieval_reject_reason_atoms": sorted(set(retrieval_reject_reason_atoms)),
    }
    promotion_attempted = bool(reuse_attempted.get("promotion_attempted"))
    promotion_used = reuse_source == "promotion"

    summary = {
        "run_id": run_dir.name,
        "task_id": task_id,
        "verdict": overall_verdict,
        "witness_path": str(witness_path),
        "ucr_path": str(ucr_path),
        "active_view_hash": active_view_hash,
        "spec_hash": spec_hash,
        "warm_start_provided": warm_start_provided,
        "warm_start_store": warm_start,
        "warm_start_candidate_hash": warm_start_candidate_hash,
        "warm_start_candidate_rejected": warm_start_candidate_rejected,
        "warm_start_reject_reason_atoms": reuse_reject_reason_atoms[
            "warm_start_reject_reason_atoms"
        ],
        "promotion_best_hash_used": promotion_best_hash_used,
        "promotion_best_tier_used": promotion_best_tier_used,
        "promotion_reject_reason_atoms": reuse_reject_reason_atoms[
            "promotion_reject_reason_atoms"
        ],
        "promotion_attempted": promotion_attempted,
        "promotion_best_hash": promotion_best_hash if promotion_attempted else "",
        "promotion_best_tier": promotion_best_tier if promotion_attempted else "",
        "promotion_used": promotion_used,
        "reuse_source": reuse_source,
        "reuse_attempted": reuse_attempted,
        "reuse_reject_reason_atoms": reuse_reject_reason_atoms,
        "proposer_error_atom": proposer_record.get("error_atom", ""),
        "proposer_kind": proposer_kind,
        "repair_kind": repair_kind,
        "repair_edits_count": repair_edits_count,
        "failure_hint_used": failure_hint_used,
        "proposer_seed_program_hash": proposer_seed_program_hash,
        "proposer_seed_source": proposer_seed_source,
        "synth_ns": synth_cost,
        "verify_ns": verify_cost,
        "breaker_ns": breaker_cost,
        "breaker_attempts": breaker_attempts,
        "meta_cases": meta_cases,
        "failure_reason": failure_reason,
    }
    if settings.expected_verdict:
        expected = str(settings.expected_verdict).upper()
        verdict_matches = overall_verdict == expected
        summary["expected_verdict"] = expected
        summary["verdict_matches_expected"] = verdict_matches
        if not verdict_matches:
            summary["expectation_mismatch_atoms"] = ["EXPECTATION_MISMATCH"]
    if program_changed is not None:
        summary["program_changed"] = program_changed
    return summary
