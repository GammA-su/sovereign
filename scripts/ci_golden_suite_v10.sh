#!/usr/bin/env bash
set -euo pipefail
OUT="runs/suite_ci_v10_golden"
rm -rf "$OUT"
uv run sovereidolon suite run --suite-file examples/suites/suite_v10.json --out-dir "$OUT"
diff -u examples/baselines/suite_v10.report.norm.json "$OUT/report.norm.json"
uv run sovereidolon store audit --store "$OUT/store"
