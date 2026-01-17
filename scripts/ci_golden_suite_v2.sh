#!/usr/bin/env bash
set -euo pipefail
OUT="runs/suite_ci_v2_golden"
rm -rf "$OUT"
uv run sovereidolon suite run --suite-file examples/suites/suite_v2.json --out-dir "$OUT"
uv run sovereidolon store audit --store "$OUT/store"
diff -u examples/baselines/suite_v2.report.norm.json "$OUT/report.norm.json"
