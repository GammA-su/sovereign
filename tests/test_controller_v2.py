from sovereidolon_v1.dominance_controller import DominanceController, dominance_v2_policy


def test_controller_v2_cost_ceiling_reject() -> None:
    controller = DominanceController(dominance_v2_policy())
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 2_000_001, "breaker_attempts": 0, "verifier_attempts": 0},
        context={"families_mode": "public", "coverage_atoms": []},
    )
    assert decision.decision == "REJECT"
    assert "COST_CEILING_EXCEEDED" in decision.reason_atoms


def test_controller_v2_coverage_reward_increases_score() -> None:
    controller = DominanceController(dominance_v2_policy())
    decision_base = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 0, "breaker_attempts": 0, "verifier_attempts": 0},
        context={"families_mode": "public", "coverage_atoms": ["lane:pyexec"]},
    )
    decision_more = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 0, "breaker_attempts": 0, "verifier_attempts": 0},
        context={
            "families_mode": "public",
            "coverage_atoms": ["lane:pyexec", "meta:commutative"],
        },
    )
    assert decision_base.score_scaled is not None
    assert decision_more.score_scaled is not None
    assert decision_more.score_scaled > decision_base.score_scaled


def test_controller_v2_sealed_weight_applies() -> None:
    controller = DominanceController(dominance_v2_policy())
    decision_public = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 0, "breaker_attempts": 0, "verifier_attempts": 0},
        context={"families_mode": "public", "coverage_atoms": []},
    )
    decision_sealed = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 0, "breaker_attempts": 0, "verifier_attempts": 0},
        context={"families_mode": "sealed", "coverage_atoms": []},
    )
    assert decision_public.score_scaled is not None
    assert decision_sealed.score_scaled is not None
    assert decision_sealed.score_scaled > decision_public.score_scaled


def test_controller_v2_reason_atoms_sorted_and_stable() -> None:
    controller = DominanceController(dominance_v2_policy())
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": False},
        breaker_kpi={},
        costs={"synth_ns": 3_000_000, "breaker_attempts": 0, "verifier_attempts": 0},
        context={
            "families_mode": "public",
            "coverage_atoms": ["lane:pyexec"],
            "metamorphic_violation": True,
        },
    )
    assert decision.decision == "REJECT"
    assert decision.reason_atoms == sorted(decision.reason_atoms)
    assert "COST_CEILING_EXCEEDED" in decision.reason_atoms
    assert "METAMORPHIC_VIOLATION" in decision.reason_atoms
