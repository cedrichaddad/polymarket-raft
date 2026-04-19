"""Build the per-second state table (§9.2).

Reads raw Parquet, joins chainlink/binance + best-bid/ask + signed trade flow
on a 1-second grid, and writes `data/derived/market_state_1s.parquet`.

Usage:
    python -m raft_research.build_state_table
"""
from __future__ import annotations
import argparse
import logging

from .duck import connect
from .paths import derived

log = logging.getLogger(__name__)


STATE_SQL = """
WITH
chain AS (
    SELECT ts_feed_ms, value AS chainlink_price
    FROM rtds_prices
    WHERE topic = 'chainlink' AND value IS NOT NULL
),
bin AS (
    SELECT ts_feed_ms, value AS binance_price
    FROM rtds_prices
    WHERE topic = 'binance' AND value IS NOT NULL
),
-- One row per (market, second): last event within that second.
book AS (
    SELECT
        market_id,
        asset_id,
        (CAST(ts_recv_local_ms / 1000 AS BIGINT)) AS state_ts_s,
        ARG_MAX(best_bid, ts_recv_local_ms) AS best_bid,
        ARG_MAX(best_ask, ts_recv_local_ms) AS best_ask
    FROM market_book_events
    GROUP BY 1, 2, 3
),
meta AS (
    SELECT
        market_id,
        ANY_VALUE(window_type) AS window_type,
        MIN(start_ts_ms) AS start_ts_ms,
        MAX(end_ts_ms) AS end_ts_ms,
        ANY_VALUE(asset_yes_id) AS asset_yes_id
    FROM market_meta
    GROUP BY 1
),
chain_s AS (
    SELECT CAST(ts_feed_ms / 1000 AS BIGINT) AS state_ts_s,
           ARG_MAX(chainlink_price, ts_feed_ms) AS chainlink_price
    FROM chain GROUP BY 1
),
bin_s AS (
    SELECT CAST(ts_feed_ms / 1000 AS BIGINT) AS state_ts_s,
           ARG_MAX(binance_price, ts_feed_ms) AS binance_price
    FROM bin GROUP BY 1
),
trades_s AS (
    SELECT
        market_id,
        CAST(ts_recv_local_ms / 1000 AS BIGINT) AS state_ts_s,
        SUM(CASE WHEN side_aggressor = 'Buy' THEN size
                 WHEN side_aggressor = 'Sell' THEN -size
                 ELSE 0 END) AS signed_flow_1s
    FROM market_trade_events
    GROUP BY 1, 2
)
SELECT
    b.market_id,
    b.state_ts_s * 1000 AS state_ts_ms,
    m.window_type,
    m.start_ts_ms,
    m.end_ts_ms,
    m.end_ts_ms - b.state_ts_s * 1000 AS tte_ms,
    c.chainlink_price,
    bn.binance_price,
    b.best_bid AS best_yes_bid,
    b.best_ask AS best_yes_ask,
    (b.best_bid + b.best_ask) / 2 AS mid_yes,
    (b.best_ask - b.best_bid) AS spread_yes,
    COALESCE(t.signed_flow_1s, 0) AS signed_flow_1s,
    (c.chainlink_price - bn.binance_price) AS chainlink_binance_gap
FROM book b
JOIN meta m ON m.market_id = b.market_id AND b.asset_id = m.asset_yes_id
LEFT JOIN chain_s c USING (state_ts_s)
LEFT JOIN bin_s bn USING (state_ts_s)
LEFT JOIN trades_s t ON t.market_id = b.market_id AND t.state_ts_s = b.state_ts_s
WHERE m.start_ts_ms <= b.state_ts_s * 1000
  AND b.state_ts_s * 1000 <= m.end_ts_ms
ORDER BY b.market_id, b.state_ts_s;
"""


def build(output: str | None = None) -> int:
    con = connect()
    # Fail loudly if the core views are missing — that means no raw data yet.
    # market_trade_events is optional (duck.py creates an empty fallback view).
    for view in ("rtds_prices", "market_book_events", "market_meta"):
        exists = con.execute(
            "SELECT COUNT(*) FROM duckdb_views() WHERE view_name = ?", [view]
        ).fetchone()[0]
        if not exists:
            raise SystemExit(f"missing view {view}: collect some data first")

    df = con.execute(STATE_SQL).df()
    out = output or str(derived("market_state_1s.parquet"))
    df.to_parquet(out, index=False)
    log.info("wrote %d rows to %s", len(df), out)
    return len(df)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output")
    args = ap.parse_args()
    build(args.output)


if __name__ == "__main__":
    _main()
