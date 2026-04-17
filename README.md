# RAFT — Resolution-Aligned Fee-Aware Trading

Implementation of the design in [docs/DESIGN.md](docs/DESIGN.md).

## Layout

```
Cargo.toml                      Rust workspace (single crate)
src/
  bin/raft.rs                   service binary
  collector/                    RTDS, market WS, REST hydrator (§7.4–§7.7)
  features/                     live state, vol, flow, state builder (§10, Appendix C)
  models/                       fair_value, calibrator, retrieval (§11, §17)
  execution/                    router, quote_manager, risk, heartbeat, broker (§13)
  storage/                      Parquet sink (§7.10, §9.1)
  event_bus.rs, health.rs, types.rs, config.rs
config/raft.yaml                Appendix B config
research/raft_research/         Python offline research + backtests (§15.2)
data/raw/                       canonical raw Parquet archive
data/derived/                   built-from-raw tables, calibrator, indexes
```

## Build order (Appendix F)

1. RTDS collector — `src/collector/rtds_ws.rs`
2. market WebSocket collector — `src/collector/market_ws.rs`
3. Parquet sinks — `src/storage/parquet_sink.rs`
4. DuckDB state-table build — `research/raft_research/build_state_table.py`
5. Baseline fair-value model — `src/models/fair_value.rs` + `research/raft_research/fair_value.py`
6. Vibrato indexing of state vectors — `research/raft_research/build_index.py` (hnswlib stand-in)
7. Retrieval overlay — `src/models/retrieval.rs`
8. Taker backtest — `research/raft_research/backtest_taker.py`
9. Maker backtest with markouts — `research/raft_research/backtest_maker.py`
10. Paper trader — `src/bin/raft.rs` in `paper` mode

## Running the service

```bash
# Research mode: connect, persist Parquet, no decisions.
cargo run --release -- run --config config/raft.yaml --mode research

# Paper mode: also spin up the decision loop against PaperBroker.
cargo run --release -- run --config config/raft.yaml --mode paper

# Live mode is intentionally guarded off — v1 has no real broker wired.
```

Unit tests:

```bash
cargo test
```

## Running the research stack

```bash
cd research
pip install -e .
python -m raft_research.build_state_table
python -m raft_research.build_labels
python -m raft_research.calibrate --train-until 2026-04-01 --test-from 2026-04-01
python -m raft_research.build_index
python -m raft_research.backtest_taker --all-in 0.012
python -m raft_research.backtest_maker --rebate-mode conservative
python -m raft_research.markouts
python -m raft_research.compare
```

## MVP scope (§23)

* BTC 5m only
* one row per second
* basic retrieval query (no subsequence search)
* isotonic calibration
* conservative maker fill model
* taker trades only in final window segment
* paper mode only — real capital is out of scope for v1

## Status

* **Phase 0** — RTDS + market WS collectors, REST hydrator, Parquet sink: implemented.
* **Phase 1** — state table build / labels: implemented in Python.
* **Phase 2** — parametric p_0 + isotonic calibration: implemented.
* **Phase 3** — retrieval overlay via hnswlib: implemented (swap in Vibrato later).
* **Phase 4** — taker + maker backtests with markouts: implemented.
* **Phase 5** — paper trader with decision loop: implemented.
* **Phase 6** — live capital: not started. `live` mode is blocked at runtime.

## What's intentionally stubbed

* Broker connectivity for real Polymarket CLOB signed orders (only `PaperBroker` in v1).
* Vibrato integration (code uses an hnswlib-backed reference implementation).
* Gamma API pagination — v1 pulls one page of active BTC minute events.
* Direct Binance / direct Chainlink ingestion (§7.12 — validation mode only).
