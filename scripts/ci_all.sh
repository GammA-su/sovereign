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
./scripts/ci_sealed.sh

junk_status=$(git status --porcelain)
if echo "$junk_status" | grep -E -q '^(\\?\\?|[ MADRCU])\\s+(runs/|store/|.*__pycache__/)'; then
  echo "CI hygiene failure: runs/, store/, or __pycache__/ contains tracked or untracked artifacts."
  echo "$junk_status" | grep -E '^(\\?\\?|[ MADRCU])\\s+(runs/|store/|.*__pycache__/)'
  exit 1
fi
