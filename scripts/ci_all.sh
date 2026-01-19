#!/usr/bin/env bash
set -euo pipefail
uv run ruff check .
uv run mypy src
uv run pytest
./scripts/ci_golden_suite.sh
./scripts/ci_golden_suite_v2.sh
./scripts/ci_golden_suite_v3.sh
./scripts/ci_golden_suite_v3_warm.sh
