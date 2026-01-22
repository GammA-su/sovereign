#!/usr/bin/env bash
set -euo pipefail

OUT="runs/learn_loop_ci_smoke"
PROMO="promoted_store/learn_loop_ci_smoke"

rm -rf "$OUT" "$PROMO"

uv run sovereidolon learn loop \
  --suite-file examples/suites/suite_v12_learn.json \
  --sealed-suite-file examples/sealed/sealed_v4_learn.json \
  --out-dir "$OUT" \
  --promo-store "$PROMO" \
  --iters 2 \
  --prefer-promotion-store \
  --proposer heuristic_v1

uv run python - <<'PY'
import json
from pathlib import Path

summary_path = Path("runs/learn_loop_ci_smoke/loop_summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))
assert data.get("schema_version") == "v3"
iterations = data.get("iterations", [])
assert iterations
last = iterations[-1]
propose = last.get("propose", {})
assert propose.get("propose_programs_total", 0) > 0
PY

for phase in iter000_public iter000_propose iter000_sealed iter001_public iter001_propose iter001_sealed; do
  uv run sovereidolon store audit --store "$OUT/$phase/store"
done
