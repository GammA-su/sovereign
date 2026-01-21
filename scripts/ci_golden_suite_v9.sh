#!/usr/bin/env bash
set -euo pipefail
OUT="runs/suite_ci_v9_golden"
rm -rf "$OUT"
uv run sovereidolon suite run --suite-file examples/suites/suite_v9.json --out-dir "$OUT"
diff -u examples/baselines/suite_v9.report.norm.json "$OUT/report.norm.json"
uv run sovereidolon store audit --store "$OUT/store"
