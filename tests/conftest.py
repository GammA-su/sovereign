import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    _ = config
    run_slow = os.getenv("RUN_SLOW_TESTS", "") or os.getenv("SOVEREIDOLON_RUN_SLOW", "")
    if str(run_slow).strip().lower() in {"1", "true", "yes"}:
        return
    skip_slow = pytest.mark.skip(reason="set RUN_SLOW_TESTS=1 to run slow tests")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
