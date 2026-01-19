#!/usr/bin/env bash
set -euo pipefail
OUT="runs/sealed_ci_v1"
rm -rf "$OUT"
uv run sovereidolon suite run-sealed --suite-file examples/sealed/sealed_v1.json --out-dir "$OUT"
uv run sovereidolon store audit --store "$OUT/store"
test -f "$OUT/sealed_summary.json"
