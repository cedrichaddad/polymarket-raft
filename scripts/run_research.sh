#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/config/raft.yaml}"

MODE="${MODE:-research}"
COLLECT_SECS="${COLLECT_SECS:-300}"
TRAIN_UNTIL="${TRAIN_UNTIL:-}"
TEST_FROM="${TEST_FROM:-}"
# Fractional split used when TRAIN_UNTIL/TEST_FROM are not set (good for short runs).
TRAIN_FRACTION="${TRAIN_FRACTION:-0.7}"

TAKER_WINDOW_SECS="${TAKER_WINDOW_SECS:-30}"
TAKER_FEE_PROB="${TAKER_FEE_PROB:-0.0}"
TAKER_ALL_IN="${TAKER_ALL_IN:-0.012}"
SIGMA_PER_SEC="${SIGMA_PER_SEC:-5e-4}"

CENTER_LOW="${CENTER_LOW:-0.40}"
CENTER_HIGH="${CENTER_HIGH:-0.60}"
MIN_DWELL_SECS="${MIN_DWELL_SECS:-2}"
MIN_OPP_VOLUME="${MIN_OPP_VOLUME:-50}"
REBATE_MODE="${REBATE_MODE:-conservative}"

RUN_COMPARE="${RUN_COMPARE:-1}"
RUN_MARKOUTS="${RUN_MARKOUTS:-1}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

RAFT_PID=""
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/raft_collector.log"

cleanup() {
  if [[ -n "${RAFT_PID}" ]] && kill -0 "${RAFT_PID}" 2>/dev/null; then
    echo "==> stopping collector (pid=${RAFT_PID})"
    kill "${RAFT_PID}" || true
    for _ in {1..10}; do
      if ! kill -0 "${RAFT_PID}" 2>/dev/null; then
        wait "${RAFT_PID}" 2>/dev/null || true
        return
      fi
      sleep 1
    done
    echo "==> collector did not exit cleanly; forcing kill -9"
    kill -9 "${RAFT_PID}" || true
    wait "${RAFT_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

run_python() {
  (
    cd "$ROOT/research"
    "$PYTHON_BIN" -m "$@"
  )
}

print_json_summary() {
  local file="$1"
  if [[ -f "$file" ]]; then
    "$PYTHON_BIN" - <<PY
import json, pathlib
p = pathlib.Path("$file")
print(json.dumps(json.loads(p.read_text()), indent=2))
PY
  else
    echo "missing: $file"
  fi
}

echo "==> repo root: $ROOT"
echo "==> config: $CONFIG"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found at $CONFIG"
  exit 1
fi

mkdir -p "$LOG_DIR"

echo "==> building rust binary"
(
  cd "$ROOT"
  cargo build --release
)

if [[ "$SKIP_INSTALL" != "1" ]]; then
  echo "==> installing research package"
  (
    cd "$ROOT/research"
    "$PYTHON_BIN" -m pip install -e .
  )
fi

if [[ "$SKIP_COLLECT" != "1" ]]; then
  echo "==> starting collector in mode=$MODE for ${COLLECT_SECS}s"
  echo "==> collector log: $LOG_FILE"
  (
    cd "$ROOT"
    cargo run --release -- run --config "$CONFIG" --mode "$MODE" \
      > "$LOG_FILE" 2>&1
  ) &
  RAFT_PID="$!"

  for ((i=0; i<COLLECT_SECS; i++)); do
    if ! kill -0 "$RAFT_PID" 2>/dev/null; then
      echo "ERROR: collector exited early; last 50 log lines:"
      tail -n 50 "$LOG_FILE" || true
      exit 1
    fi
    sleep 1
  done

  cleanup
  RAFT_PID=""
else
  echo "==> SKIP_COLLECT=1, reusing existing data"
fi

echo "==> building state table"
run_python raft_research.build_state_table

echo "==> building labels"
run_python raft_research.build_labels

echo "==> calibrating"
if [[ -n "$TRAIN_UNTIL" && -n "$TEST_FROM" ]]; then
  run_python raft_research.calibrate \
    --train-until "$TRAIN_UNTIL" \
    --test-from "$TEST_FROM" \
    --sigma-per-sec "$SIGMA_PER_SEC"
else
  run_python raft_research.calibrate \
    --fraction "$TRAIN_FRACTION" \
    --sigma-per-sec "$SIGMA_PER_SEC"
fi

echo "==> building retrieval index"
run_python raft_research.build_index

echo "==> taker backtest"
run_python raft_research.backtest_taker \
  --taker-window-secs "$TAKER_WINDOW_SECS" \
  --taker-fee-prob "$TAKER_FEE_PROB" \
  --all-in "$TAKER_ALL_IN" \
  --sigma-per-sec "$SIGMA_PER_SEC" \
  --calibrator "$ROOT/data/derived/calibrator.json"

echo "==> maker backtest"
run_python raft_research.backtest_maker \
  --center-low "$CENTER_LOW" \
  --center-high "$CENTER_HIGH" \
  --min-dwell-secs "$MIN_DWELL_SECS" \
  --min-opp-volume "$MIN_OPP_VOLUME" \
  --rebate-mode "$REBATE_MODE"

if [[ "$RUN_MARKOUTS" == "1" ]]; then
  echo "==> markout aggregation"
  run_python raft_research.markouts
fi

if [[ "$RUN_COMPARE" == "1" ]]; then
  echo "==> model comparison"
  run_python raft_research.compare --sigma-per-sec "$SIGMA_PER_SEC"
fi

echo
echo "==> taker summary"
print_json_summary "$ROOT/data/derived/backtest_taker/summary.json"

echo
echo "==> maker summary"
print_json_summary "$ROOT/data/derived/backtest_maker/summary.json"

echo
echo "==> outputs"
echo "  data/derived/backtest_taker/summary.json"
echo "  data/derived/backtest_taker/trades.parquet"
echo "  data/derived/backtest_maker/summary.json"
echo "  data/derived/backtest_maker/fills.parquet"
echo "  data/derived/markouts_aggregate.csv"
echo "  data/derived/compare_models.csv"
echo "  logs/raft_collector.log"