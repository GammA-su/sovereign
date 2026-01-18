#!/usr/bin/env bash
set -euo pipefail
uv run ruff check .
uv run mypy src
uv run pytest
./scripts/ci_golden_suite.sh
./scripts/ci_golden_suite_v2.sh
./scripts/ci_golden_suite_v3.sh
./scripts/ci_golden_suite_v3_warm.sh
./scripts/ci_golden_suite_v4.sh
./scripts/ci_golden_suite_v5.sh
./scripts/ci_golden_suite_v6.sh
./scripts/ci_golden_suite_v6_warm.sh
./scripts/ci_golden_suite_v7.sh

tracked_junk=$(git ls-files | rg -n "^(runs/|store/|.*__pycache__/)" || true)
if [[ -n "${tracked_junk}" ]]; then
  echo "Tracked junk detected:"
  echo "${tracked_junk}"
  exit 1
fi
dirty_junk=$(git status --porcelain --untracked-files=all | rg -n "^(\\?\\?| M|A) (runs/|store/|.*__pycache__/)" || true)
if [[ -n "${dirty_junk}" ]]; then
  echo "Junk artifacts detected in working tree:"
  echo "${dirty_junk}"
  exit 1
fi
