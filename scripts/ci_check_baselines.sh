#!/usr/bin/env bash
set -euo pipefail

fail=0

run_golden() {
  local name="$1"
  local suite_file="$2"
  local baseline="$3"
  local out_dir="$4"
  local proposer_cmd="$5"

  rm -rf "$out_dir"
  if [[ -n "$proposer_cmd" ]]; then
    uv run sovereidolon suite run \
      --suite-file "$suite_file" \
      --out-dir "$out_dir" \
      --proposer subprocess \
      --proposer-cmd "$proposer_cmd"
  else
    uv run sovereidolon suite run --suite-file "$suite_file" --out-dir "$out_dir"
  fi

  if ! diff -u "$baseline" "$out_dir/report.norm.json"; then
    echo "baseline drift: $name"
    echo "fix with: cp \"$out_dir/report.norm.json\" \"$baseline\""
    fail=1
  fi
  uv run sovereidolon store audit --store "$out_dir/store"
}

run_golden \
  "suite_v10" \
  "examples/suites/suite_v10.json" \
  "examples/baselines/suite_v10.report.norm.json" \
  "runs/suite_ci_v10_golden" \
  ""

REPO_ROOT="$(pwd)"
run_golden \
  "suite_v11_subprocess" \
  "examples/suites/suite_v11_subprocess.json" \
  "examples/baselines/suite_v11_subprocess.report.norm.json" \
  "runs/suite_ci_v11_subprocess_golden" \
  "uv run python ${REPO_ROOT}/tests/fixtures/proposer_echo.py"

if [[ $fail -ne 0 ]]; then
  exit 1
fi
