#!/usr/bin/env bash
set -euo pipefail
OUT_A="runs/suite_ci_v6_warm_a"
OUT_B="runs/suite_ci_v6_warm_b"
rm -rf "$OUT_A" "$OUT_B"
uv run sovereidolon suite run --suite-file examples/suites/suite_v6.json --out-dir "$OUT_A"
uv run sovereidolon suite run --suite-file examples/suites/suite_v6.json --out-dir "$OUT_B" --warm-start-store "$OUT_A/store"
uv run sovereidolon store audit --store "$OUT_B/store"
diff -u examples/baselines/suite_v6_warm.report.norm.json "$OUT_B/report.norm.json"
