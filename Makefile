SHELL := /bin/bash

PYTHON ?= python3
CONFIG ?= config/raft.yaml

TRAIN_UNTIL ?=
TEST_FROM ?=
TRAIN_FRACTION ?= 0.7

COLLECT_SECS ?= 300
MODE ?= research

TAKER_WINDOW_SECS ?= 30
TAKER_FEE_PROB ?= 0.0
TAKER_ALL_IN ?= 0.012
SIGMA_PER_SEC ?= 5e-4

CENTER_LOW ?= 0.40
CENTER_HIGH ?= 0.60
MIN_DWELL_SECS ?= 2
MIN_OPP_VOLUME ?= 50
REBATE_MODE ?= conservative

SKIP_INSTALL ?= 0
SKIP_COLLECT ?= 0

.PHONY: help install build print-config collect paper build-state labels calibrate index \
        backtest-taker backtest-maker markouts compare run-research sweep summaries clean chmod-scripts \
        sizing-ablation

help:
	@echo "Targets:"
	@echo "  make install           - install research package"
	@echo "  make build             - cargo build --release"
	@echo "  make print-config      - print resolved config"
	@echo "  make collect           - run Rust collector in research mode"
	@echo "  make paper             - run Rust collector in paper mode"
	@echo "  make build-state       - build market_state_1s.parquet"
	@echo "  make labels            - build labeled state table"
	@echo "  make calibrate         - fit calibrator"
	@echo "  make index             - build hnswlib retrieval index"
	@echo "  make backtest-taker    - run taker backtest"
	@echo "  make backtest-maker    - run maker backtest"
	@echo "  make markouts          - aggregate maker markouts"
	@echo "  make compare           - compare p_0 / p_star / p_hybrid"
	@echo "  make run-research      - end-to-end collector + pipeline"
	@echo "  make sweep             - parameter sweep for taker + maker"
	@echo "  make sizing-ablation   - rolling-holdout bucketed-sizing ablation"
	@echo "  make summaries         - print current summaries"
	@echo "  make chmod-scripts     - make helper scripts executable"
	@echo "  make clean             - remove derived outputs"

install:
	cd research && $(PYTHON) -m pip install -e .

build:
	cargo build --release

print-config:
	cargo run --release -- print-config --config $(CONFIG)

collect:
	cargo run --release -- run --config $(CONFIG) --mode research

paper:
	cargo run --release -- run --config $(CONFIG) --mode paper

build-state:
	cd research && $(PYTHON) -m raft_research.build_state_table

labels:
	cd research && $(PYTHON) -m raft_research.build_labels

calibrate:
	cd research && $(PYTHON) -m raft_research.calibrate \
		$(if $(TRAIN_UNTIL),--train-until $(TRAIN_UNTIL) --test-from $(TEST_FROM),--fraction $(TRAIN_FRACTION)) \
		--sigma-per-sec $(SIGMA_PER_SEC)

index:
	cd research && $(PYTHON) -m raft_research.build_index

backtest-taker:
	cd research && $(PYTHON) -m raft_research.backtest_taker \
		--taker-window-secs $(TAKER_WINDOW_SECS) \
		--taker-fee-prob $(TAKER_FEE_PROB) \
		--all-in $(TAKER_ALL_IN) \
		--sigma-per-sec $(SIGMA_PER_SEC) \
		--calibrator ../data/derived/calibrator.json

backtest-maker:
	cd research && $(PYTHON) -m raft_research.backtest_maker \
		--center-low $(CENTER_LOW) \
		--center-high $(CENTER_HIGH) \
		--min-dwell-secs $(MIN_DWELL_SECS) \
		--min-opp-volume $(MIN_OPP_VOLUME) \
		--rebate-mode $(REBATE_MODE)

markouts:
	cd research && $(PYTHON) -m raft_research.markouts

compare:
	cd research && $(PYTHON) -m raft_research.compare --sigma-per-sec $(SIGMA_PER_SEC)

run-research:
	COLLECT_SECS=$(COLLECT_SECS) \
	MODE=$(MODE) \
	TRAIN_UNTIL=$(TRAIN_UNTIL) \
	TEST_FROM=$(TEST_FROM) \
	TRAIN_FRACTION=$(TRAIN_FRACTION) \
	TAKER_WINDOW_SECS=$(TAKER_WINDOW_SECS) \
	TAKER_FEE_PROB=$(TAKER_FEE_PROB) \
	TAKER_ALL_IN=$(TAKER_ALL_IN) \
	SIGMA_PER_SEC=$(SIGMA_PER_SEC) \
	CENTER_LOW=$(CENTER_LOW) \
	CENTER_HIGH=$(CENTER_HIGH) \
	MIN_DWELL_SECS=$(MIN_DWELL_SECS) \
	MIN_OPP_VOLUME=$(MIN_OPP_VOLUME) \
	REBATE_MODE=$(REBATE_MODE) \
	SKIP_INSTALL=$(SKIP_INSTALL) \
	SKIP_COLLECT=$(SKIP_COLLECT) \
	PYTHON_BIN=$(PYTHON) \
	./scripts/run_research.sh

sweep:
	PYTHON_BIN=$(PYTHON) ./scripts/sweep_backtests.sh

sizing-ablation:
	cd research && $(PYTHON) -m raft_research.sizing_ablation \
		--calibrator ../data/derived/calibrator.json \
		--taker-fee-prob $(TAKER_FEE_PROB) \
		--n-folds 4

summaries:
	@echo "==> taker"
	@cat data/derived/backtest_taker/summary.json 2>/dev/null || true
	@echo
	@echo "==> maker"
	@cat data/derived/backtest_maker/summary.json 2>/dev/null || true

chmod-scripts:
	chmod +x scripts/run_research.sh scripts/sweep_backtests.sh

clean:
	rm -rf data/derived/backtest_taker \
	       data/derived/backtest_maker \
	       data/derived/sweeps \
	       data/derived/compare_models.csv \
	       data/derived/markouts_aggregate.csv