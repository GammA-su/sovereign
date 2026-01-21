#!/usr/bin/env bash
set -euo pipefail
PROMO="promoted_store/v1"
OUT_A="runs/ci_promo_a"
OUT_B="runs/ci_promo_b"
rm -rf "$OUT_A" "$OUT_B" "$PROMO"
uv run sovereidolon suite run --suite-file examples/suites/suite_v6.json --out-dir "$OUT_A" --promotion-store "$PROMO"
uv run sovereidolon suite run --suite-file examples/suites/suite_v6.json --out-dir "$OUT_B" --promotion-store "$PROMO" --prefer-promotion-store
uv run sovereidolon store audit --store "$OUT_A/store"
uv run sovereidolon store audit --store "$OUT_B/store"
test -f "$PROMO/index.json"
python -c 'import json; data=json.load(open("runs/ci_promo_b/report.json")); tasks={t["task_id"]:t for t in data.get("per_task",[])}; [(lambda t: t and t.get("verdict")=="PASS" and t.get("synth_ns")==0 and t.get("warm_start_store") is True and t.get("warm_start_candidate_hash")==t.get("program_hash") or (_ for _ in ()).throw(AssertionError(tid)))(tasks.get(tid)) for tid in ("codepatch_pass_01","pyfunc_01")]'
