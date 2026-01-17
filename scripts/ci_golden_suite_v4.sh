#!/usr/bin/env bash
set -euo pipefail
OUT="runs/suite_ci_v4_golden"
rm -rf "$OUT"
uv run sovereidolon suite run --suite-file examples/suites/suite_v4.json --out-dir "$OUT"
uv run sovereidolon store audit --store "$OUT/store"
diff -u examples/baselines/suite_v4.report.norm.json "$OUT/report.norm.json"
