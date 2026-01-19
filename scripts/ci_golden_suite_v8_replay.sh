#!/usr/bin/env bash
set -euo pipefail
OUT_RECORD="runs/suite_ci_v8_record"
OUT_REPLAY="runs/suite_ci_v8_replay"
REPLAY_FILE="$OUT_RECORD/proposals.jsonl"
rm -rf "$OUT_RECORD" "$OUT_REPLAY"
uv run sovereidolon suite run --suite-file examples/suites/suite_v8.json --out-dir "$OUT_RECORD" --record-proposals "$REPLAY_FILE"
uv run sovereidolon suite run --suite-file examples/suites/suite_v8.json --out-dir "$OUT_REPLAY" --proposer replay --replay-file "$REPLAY_FILE"
diff -u examples/baselines/suite_v8_replay.report.norm.json "$OUT_REPLAY/report.norm.json"
uv run sovereidolon store audit --store "$OUT_REPLAY/store"
