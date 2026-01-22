#!/usr/bin/env bash
set -euo pipefail
OUT="runs/suite_ci_v11_subprocess_golden"
rm -rf "$OUT"
REPO_ROOT="$(pwd)"
uv run sovereidolon suite run \
  --suite-file examples/suites/suite_v11_subprocess.json \
  --out-dir "$OUT" \
  --proposer subprocess \
  --proposer-cmd "uv run python ${REPO_ROOT}/tests/fixtures/proposer_echo.py"
diff -u examples/baselines/suite_v11_subprocess.report.norm.json "$OUT/report.norm.json"
uv run sovereidolon store audit --store "$OUT/store"
