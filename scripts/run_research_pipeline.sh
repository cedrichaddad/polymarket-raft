#!/usr/bin/env bash
# End-to-end offline research pipeline.
# Assumes raw Parquet already exists under data/raw/.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m raft_research.build_state_table
python -m raft_research.build_labels
python -m raft_research.calibrate --train-until "${TRAIN_UNTIL:-2026-04-01}" --test-from "${TEST_FROM:-2026-04-01}"
python -m raft_research.build_index
python -m raft_research.backtest_taker --all-in "${ALL_IN:-0.012}"
python -m raft_research.backtest_maker --rebate-mode "${REBATE_MODE:-conservative}"
python -m raft_research.markouts
python -m raft_research.compare
