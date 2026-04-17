"""DuckDB connection factory + common views over the raw Parquet archive.

The Rust service writes events with a uniform 8-column schema
(ts_exchange_ms, ts_recv_local_ms, f1, f2, s1, s2, s3, body_json). This module
exposes typed views so downstream queries are readable.
"""
from __future__ import annotations
import duckdb
from pathlib import Path
from .paths import RAW_ROOT


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:", read_only=False)
    _install_views(con)
    return con


def _install_views(con: duckdb.DuckDBPyConnection) -> None:
    root = RAW_ROOT.resolve()
    if not root.exists():
        return

    # rtds: s1=topic, s2=symbol, f1=value
    rtds_glob = _glob(root, "rtds")
    if rtds_glob:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW rtds_prices AS
            SELECT
                ts_exchange_ms AS ts_feed_ms,
                ts_recv_local_ms,
                s1 AS topic,
                s2 AS symbol,
                f1 AS value
            FROM read_parquet('{rtds_glob}', union_by_name=true);
            """
        )

    # market_ws: s1=market_id, s2=asset_id, s3=event_type, f1=best_bid, f2=best_ask
    mkt_glob = _glob(root, "market_ws")
    if mkt_glob:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW market_book_events AS
            SELECT
                ts_exchange_ms,
                ts_recv_local_ms,
                s1 AS market_id,
                s2 AS asset_id,
                s3 AS event_type,
                f1 AS best_bid,
                f2 AS best_ask,
                body_json
            FROM read_parquet('{mkt_glob}', union_by_name=true);
            """
        )

    # trades
    trades_glob = _glob(root, "trades")
    if trades_glob:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW market_trade_events AS
            SELECT
                ts_exchange_ms,
                ts_recv_local_ms,
                s1 AS market_id,
                s2 AS asset_id,
                s3 AS side_aggressor,
                f1 AS price,
                f2 AS size,
                body_json AS trade_id
            FROM read_parquet('{trades_glob}', union_by_name=true);
            """
        )

    # meta
    meta_glob = _glob(root, "meta")
    if meta_glob:
        con.execute(
            f"""
            CREATE OR REPLACE VIEW market_meta AS
            SELECT
                s1 AS market_id,
                s2 AS window_type,
                s3 AS asset_yes_id,
                CAST(f1 AS BIGINT) AS start_ts_ms,
                CAST(f2 AS BIGINT) AS end_ts_ms,
                body_json AS fee_schedule_json,
                ts_recv_local_ms
            FROM read_parquet('{meta_glob}', union_by_name=true);
            """
        )


def _glob(root: Path, source: str) -> str | None:
    p = root / f"source={source}"
    if not p.exists():
        return None
    return str(p / "**/*.parquet")
