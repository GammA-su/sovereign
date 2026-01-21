from sovereidolon_v1.dominance_controller import DominanceController, dominance_v3_policy


def test_controller_v3_low_score_rejects() -> None:
    policy = dominance_v3_policy()
    policy["min_score_scaled"] = 1000
    policy["default_cost_ceiling_ns"] = 100_000_000
    controller = DominanceController(policy)
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 10_000_000, "verify_ns": 0, "breaker_ns": 0},
        context={"coverage_gain": 0},
    )
    assert decision.decision == "REJECT"
    assert "SCORE_TOO_LOW" in decision.reason_atoms


def test_controller_v3_high_score_admits() -> None:
    controller = DominanceController(dominance_v3_policy())
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 1, "verify_ns": 0, "breaker_ns": 0},
        context={"coverage_gain": 5, "metamorphic_pass": True},
    )
    assert decision.decision == "ADMIT"
    assert "CONTROLLER_OK" in decision.reason_atoms


def test_controller_v3_insufficient_coverage_rejects() -> None:
    controller = DominanceController(dominance_v3_policy())
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={},
        costs={"synth_ns": 1, "verify_ns": 0, "breaker_ns": 0},
        context={"coverage_gain": 0},
    )
    assert decision.decision == "REJECT"
    assert "INSUFFICIENT_COVERAGE_GAIN" in decision.reason_atoms


def test_controller_v3_policy_id_deterministic() -> None:
    policy = dominance_v3_policy()
    controller_a = DominanceController(policy)
    controller_b = DominanceController(policy)
    assert controller_a.policy_id == controller_b.policy_id


def test_controller_v3_reason_atoms_sorted() -> None:
    controller = DominanceController(dominance_v3_policy())
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": False},
        breaker_kpi={},
        costs={"synth_ns": 20_000_000, "verify_ns": 0, "breaker_ns": 0},
        context={"coverage_gain": 0, "metamorphic_violation": True},
    )
    assert decision.reason_atoms == sorted(decision.reason_atoms)
    assert "REQUIRED_LANES_FAILED" in decision.reason_atoms
