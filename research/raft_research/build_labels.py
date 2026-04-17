"""Attach terminal-resolution labels to the state table (§18.1).

For each row we look up the Chainlink print at (or just before) the market's
`end_ts_ms` and set `resolved_up = 1` iff that terminal price >= the opening
reference. The opening reference is defined as the Chainlink print at (or
just after) `start_ts_ms`.

Usage:
    python -m raft_research.build_labels
"""
from __future__ import annotations
import argparse
import logging

import numpy as np
import pandas as pd

from .duck import connect
from .paths import derived

log = logging.getLogger(__name__)


def build(input_path: str | None = None, output_path: str | None = None) -> int:
    con = connect()
    state = pd.read_parquet(input_path or str(derived("market_state_1s.parquet")))
    # Chainlink frame indexed by feed ms (ascending) — used for both open and close lookups.
    chain = con.execute(
        """
        SELECT ts_feed_ms AS ts, value AS v
        FROM rtds_prices
        WHERE topic = 'chainlink' AND value IS NOT NULL
        ORDER BY ts_feed_ms
        """
    ).df()
    if chain.empty:
        raise SystemExit("no chainlink data — cannot compute labels")

    chain_ts = chain["ts"].to_numpy()
    chain_v = chain["v"].to_numpy()

    # Per-market open and close Chainlink prints.
    per_market = state.groupby("market_id").agg(
        start_ts_ms=("start_ts_ms", "max"),
        end_ts_ms=("end_ts_ms", "max"),
    ).reset_index()

    per_market["open_ref_price"] = per_market["start_ts_ms"].apply(
        lambda ts: _lookup_fwd(chain_ts, chain_v, ts)
    )
    per_market["close_ref_price"] = per_market["end_ts_ms"].apply(
        lambda ts: _lookup_bwd(chain_ts, chain_v, ts)
    )
    per_market["resolved_up"] = (
        per_market["close_ref_price"] >= per_market["open_ref_price"]
    ).astype(int)
    per_market = per_market[
        per_market["open_ref_price"].notna() & per_market["close_ref_price"].notna()
    ]

    merged = state.merge(
        per_market[["market_id", "open_ref_price", "close_ref_price", "resolved_up"]],
        on="market_id", how="left",
    )
    out = output_path or str(derived("market_state_1s_labeled.parquet"))
    merged.to_parquet(out, index=False)
    log.info("wrote %d rows (%d markets labeled) to %s",
             len(merged), per_market["resolved_up"].notna().sum(), out)
    return len(merged)


def _lookup_fwd(ts_arr: np.ndarray, v_arr: np.ndarray, target_ms: int):
    """Return the first value with ts >= target."""
    idx = np.searchsorted(ts_arr, target_ms, side="left")
    if idx >= len(ts_arr):
        return np.nan
    return float(v_arr[idx])


def _lookup_bwd(ts_arr: np.ndarray, v_arr: np.ndarray, target_ms: int):
    """Return the last value with ts <= target."""
    idx = np.searchsorted(ts_arr, target_ms, side="right") - 1
    if idx < 0:
        return np.nan
    return float(v_arr[idx])


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--output")
    args = ap.parse_args()
    build(args.input, args.output)


if __name__ == "__main__":
    _main()
