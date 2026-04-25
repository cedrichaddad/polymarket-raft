"""Maker-side bucket diagnostics (companion to taker_gate_diagnostics).

Joins `data/derived/backtest_maker/fills.parquet` with per-market
`end_ts_ms` from the labeled state table, derives `tte_ms_at_fill` and
`tte_bucket`, and reports per-bucket PnL stats.

Writes `data/derived/maker_bucket_stats.csv`.

Usage:
    python -m raft_research.maker_bucket_diagnostics
"""
from __future__ import annotations
import logging

import numpy as np
import pandas as pd

from .paths import derived
from .taker_gate import _bucket_tte

log = logging.getLogger(__name__)


def run() -> pd.DataFrame:
    fills_p = derived("backtest_maker") / "fills.parquet"
    if not fills_p.exists():
        raise SystemExit(f"missing {fills_p} — run backtest_maker first")
    fills = pd.read_parquet(fills_p)
    if fills.empty:
        raise SystemExit("fills.parquet is empty — no maker fills to diagnose")

    state = pd.read_parquet(derived("market_state_1s_labeled.parquet"))
    end_ts = state.groupby("market_id")["end_ts_ms"].max().reset_index()
    fills = fills.merge(end_ts, on="market_id", how="left")
    fills["tte_ms_at_fill"] = (fills["end_ts_ms"] - fills["fill_ts_ms"]).clip(lower=0)
    fills["tte_s"] = fills["tte_ms_at_fill"] / 1000.0
    fills["tte_bucket"] = _bucket_tte(fills["tte_s"])

    grp = fills.groupby("tte_bucket", dropna=False)
    out = grp.agg(
        n=("net_pnl", "size"),
        total_net_pnl=("net_pnl", "sum"),
        net_pnl_mean=("net_pnl", "mean"),
        net_pnl_std=("net_pnl", "std"),
        markout_500ms_mean=("markout_500ms", "mean"),
        markout_2000ms_mean=("markout_2000ms", "mean"),
        markout_5000ms_mean=("markout_5000ms", "mean"),
        tte_s_mean=("tte_s", "mean"),
    ).reset_index()
    out["t_stat"] = out["net_pnl_mean"] / (out["net_pnl_std"] / np.sqrt(out["n"]))

    p_out = derived("maker_bucket_stats.csv")
    out.to_csv(p_out, index=False)
    log.info("wrote %s", p_out)
    log.info("\n%s", out.to_string(index=False))
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    _main()
