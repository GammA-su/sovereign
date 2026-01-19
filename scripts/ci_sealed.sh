#!/usr/bin/env bash
set -euo pipefail
OUT_V1="runs/sealed_ci_v1"
OUT_V2="runs/sealed_ci_v2"
rm -rf "$OUT_V1" "$OUT_V2"
uv run sovereidolon suite run-sealed --suite-file examples/sealed/sealed_v1.json --out-dir "$OUT_V1"
uv run sovereidolon store audit --store "$OUT_V1/store"
test -f "$OUT_V1/sealed_summary.json"
uv run sovereidolon suite run-sealed --suite-file examples/sealed/sealed_v2.json --out-dir "$OUT_V2"
uv run sovereidolon store audit --store "$OUT_V2/store"
test -f "$OUT_V2/sealed_summary.json"
