#!/usr/bin/env bash
# Full answer-level run, meant for DeepSeek's off-peak price window.
#
#   scripts/run_offpeak.sh            # refuses to start outside the window
#   scripts/run_offpeak.sh --force    # start anyway
#
# Steps: dev split x1 (stop if any error) -> test split x3 -> docs/results_test.md.
# Re-running the script resumes where it stopped (finished scenarios are reused).
# Settings come from .env (see .env.example). Window: BB_OFFPEAK, default 00:30-08:30,
# Asia/Shanghai time; check DeepSeek's current pricing page if in doubt.
set -euo pipefail
cd "$(dirname "$0")/.."

WINDOW="${BB_OFFPEAK:-00:30-08:30}"
DEV_LABEL="${BB_DEV_LABEL:-dev-full}"
TEST_LABEL="${BB_TEST_LABEL:-test-v1}"
PY=.venv/bin/python
export PYTHONPATH=src:bench MEM0_TELEMETRY=False

if [[ "${1:-}" != "--force" ]]; then
  now=$(TZ=Asia/Shanghai date +%H%M)
  start=${WINDOW%-*}; end=${WINDOW#*-}
  start=${start/:/}; end=${end/:/}
  if (( 10#$now < 10#$start || 10#$now >= 10#$end )); then
    echo "outside the off-peak window $WINDOW (Asia/Shanghai, now $(TZ=Asia/Shanghai date +%H:%M)); use --force to run anyway" >&2
    exit 2
  fi
fi

mkdir -p results
log="results/offpeak-$(date +%Y%m%d-%H%M).log"
echo "logging to $log" >&2
exec > >(tee -a "$log") 2>&1

$PY -m branchbench check-frozen --split test

resume() { [[ -d "results/$1" ]] && echo --resume || true; }

echo "== dev split, 1 run =="
$PY -m branchbench answer-run --split dev --runs 1 --label "$DEV_LABEL" $(resume "$DEV_LABEL")
errors=$($PY -c "import json,sys; print(sum(1 for l in open('results/$DEV_LABEL/run1/answers.jsonl') if json.loads(l)['error']))")
if (( errors > 0 )); then
  echo "dev run had $errors errors; not starting the test split (inspect results/$DEV_LABEL, then rerun this script)" >&2
  exit 1
fi

echo "== test split, 3 runs =="
$PY -m branchbench answer-run --split test --runs 3 --label "$TEST_LABEL" $(resume "$TEST_LABEL")
$PY -m branchbench answer-report "results/$TEST_LABEL" --markdown docs/results_test.md
echo "done: docs/results_test.md"
