"""Markout analysis (§14.5).

Reads `data/derived/backtest_maker/fills.parquet` and aggregates post-fill
markouts sliced by:
    * center vs wing
    * volatility bucket
    * spread bucket
    * time-to-expiry bucket

Writes a wide CSV to `data/derived/markouts_aggregate.csv`.
"""
from __future__ import annotations
import argparse
import logging

import numpy as np
import pandas as pd

from .features import add_features
from .paths import derived

log = logging.getLogger(__name__)

HORIZONS = [250, 500, 1000, 2000, 5000]


def analyze() -> pd.DataFrame:
    fills_path = derived("backtest_maker") / "fills.parquet"
    if not fills_path.exists():
        raise SystemExit(f"missing {fills_path} — run backtest_maker first")
    fills = pd.read_parquet(fills_path)

    state = add_features(pd.read_parquet(derived("market_state_1s_labeled.parquet")))
    # Match fills to the bar they happened in for context columns.
    join = fills.merge(
        state[["market_id", "state_ts_ms", "spread_yes", "rv_30s", "tte_ms", "tick_regime"]]
        .rename(columns={"state_ts_ms": "fill_ts_ms"}),
        on=["market_id", "fill_ts_ms"], how="left",
    )
    def _safe_qcut(series: pd.Series, q: int, labels: list[str]) -> pd.Series:
        """qcut that falls back to 'all' when there aren't enough distinct values."""
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates="drop")
        except ValueError:
            return pd.Series("all", index=series.index)

    join["vol_bucket"] = _safe_qcut(join["rv_30s"].fillna(0), q=3, labels=["low", "med", "high"])
    join["spread_bucket"] = _safe_qcut(join["spread_yes"].fillna(0), q=3, labels=["tight", "mid", "wide"])
    join["tte_bucket"] = _safe_qcut(join["tte_ms"].fillna(0), q=3, labels=["late", "mid", "early"])
    join["center_or_wing"] = np.where(join["tick_regime"] == 1.0, "center", "wing")

    rows = []
    group_cols = ["center_or_wing", "vol_bucket", "spread_bucket", "tte_bucket"]
    for keys, g in join.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n"] = int(len(g))
        for h in HORIZONS:
            col = f"markout_{h}ms"
            if col in g:
                row[f"{col}_mean"] = float(g[col].mean())
        rows.append(row)
    agg = pd.DataFrame(rows)
    out = derived("markouts_aggregate.csv")
    agg.to_csv(out, index=False)
    log.info("wrote markout aggregate to %s", out)
    return agg


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    argparse.ArgumentParser().parse_args()
    analyze()


if __name__ == "__main__":
    _main()
