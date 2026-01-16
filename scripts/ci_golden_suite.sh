#!/usr/bin/env bash
set -euo pipefail
OUT="runs/suite_ci_golden"
rm -rf "$OUT"
uv run sovereidolon suite run --suite-file examples/suites/suite_v1.json --out-dir "$OUT"
diff -u examples/baselines/suite_v1.report.norm.json "$OUT/report.norm.json"
