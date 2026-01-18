from __future__ import annotations

from typing import List

from ..schemas import VerifierVerdict
from .lanes import (
    VerifierContext,
    anchor_lane,
    codepatch_lane,
    codepatch_metamorphic_lane,
    consequence_lane,
    pyexec_lane,
    pyexec_metamorphic_lane,
    recompute_lane,
    transfer_lane,
    translation_lane,
)


def run_verifiers(ctx: VerifierContext) -> List[VerifierVerdict]:
    if ctx.task.task_type == "pyfunc":
        lanes = [pyexec_lane, pyexec_metamorphic_lane]
    elif ctx.task.task_type == "codepatch":
        lanes = [codepatch_lane, codepatch_metamorphic_lane]
    else:
        lanes = [recompute_lane, consequence_lane, translation_lane, anchor_lane, transfer_lane]
    return [lane(ctx) for lane in lanes]


def required_lanes_passed(verdicts: List[VerifierVerdict], required: List[str]) -> bool:
    verdict_by_tier = {verdict.tier: verdict for verdict in verdicts}
    for lane in required:
        verdict = verdict_by_tier.get(lane)
        if verdict is None or verdict.verdict != "PASS":
            return False
        if verdict.soundness_grade == "HEURISTIC":
            return False
    return True
