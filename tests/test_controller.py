from sovereidolon_v1.dominance_controller import DominanceController


def test_controller_rejects_kpi_minima() -> None:
    policy = {
        "policy_id": "dominance_v1",
        "default": {
            "kpi_minima": {"CDR": 0.0, "TMR": 0.0, "NOVN": 0.0, "WFHR": 0.0},
            "cost_ceilings": {"breaker_attempts": 10**6, "meta_cases": 10**6},
        },
        "domains": {
            "pyfunc": {
                "kpi_minima": {"CDR": 0.0, "TMR": 0.0, "NOVN": 1.0, "WFHR": 0.0},
                "cost_ceilings": {"breaker_attempts": 10**6, "meta_cases": 10**6},
            }
        },
    }
    controller = DominanceController(policy)
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={"NOVN": 0.0, "CDR": 0.0, "WFHR": 0.0, "TMR": 0.0},
        costs={"breaker_attempts": 0, "meta_cases": 0},
        context={},
    )
    assert decision.decision == "REJECT"
    assert "KPI_FAIL:NOVN" in decision.reason_atoms


def test_controller_rejects_cost_ceiling() -> None:
    policy = {
        "policy_id": "dominance_v1",
        "default": {
            "kpi_minima": {"CDR": 0.0, "TMR": 0.0, "NOVN": 0.0, "WFHR": 0.0},
            "cost_ceilings": {"breaker_attempts": 10, "meta_cases": 10},
        },
        "domains": {},
    }
    controller = DominanceController(policy)
    decision = controller.evaluate(
        task_domain="bvps",
        lane_results={"required_passed": True},
        breaker_kpi={"NOVN": 0.0, "CDR": 0.0, "WFHR": 0.0, "TMR": 0.0},
        costs={"breaker_attempts": 11, "meta_cases": 0},
        context={},
    )
    assert decision.decision == "REJECT"
    assert "COST_CEILING" in decision.reason_atoms


def test_controller_rejects_sealed_regression() -> None:
    policy = {
        "policy_id": "dominance_v1",
        "default": {
            "kpi_minima": {"CDR": 0.0, "TMR": 0.0, "NOVN": 0.0, "WFHR": 0.0},
            "cost_ceilings": {"breaker_attempts": 10**6, "meta_cases": 10**6},
        },
        "domains": {},
    }
    controller = DominanceController(policy)
    decision = controller.evaluate(
        task_domain="pyfunc",
        lane_results={"required_passed": True},
        breaker_kpi={"NOVN": 1.0, "CDR": 0.0, "WFHR": 1.0, "TMR": 1.0},
        costs={"breaker_attempts": 0, "meta_cases": 0},
        context={"sealed_regression": True},
    )
    assert decision.decision == "REJECT"
    assert "SEALED_REGRESSION" in decision.reason_atoms
