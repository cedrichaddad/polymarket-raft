"""Diagnostics over a finished taker backtest.

Reads `data/derived/backtest_taker/trades.parquet` and produces:

  * `data/derived/taker_gate_bucket_stats.csv`:
      per-`tte_bucket` trade count, PnL sum, mean, hit rate.
  * `data/derived/taker_gate_edge_deciles.csv`:
      PnL per decile of `abs_edge`.
  * `data/derived/taker_gate_persistence_stats.csv`:
      PnL broken down by `edge_persist_2s` and `edge_persist_3s`.

If matplotlib is available, also writes three PNGs:
  * `trade_count_by_bucket.png`
  * `avg_pnl_by_bucket.png`
  * `cumulative_pnl.png`

Plots are optional — missing matplotlib is logged and ignored.

Usage:
    python -m raft_research.taker_gate_diagnostics
"""
from __future__ import annotations
import argparse
import logging

import numpy as np
import pandas as pd

from .paths import derived

log = logging.getLogger(__name__)


def _load_trades() -> pd.DataFrame:
    p = derived("backtest_taker") / "trades.parquet"
    if not p.exists():
        raise SystemExit(f"missing {p} — run backtest_taker first")
    df = pd.read_parquet(p)
    if df.empty:
        raise SystemExit("trades.parquet is empty — no trades to diagnose")
    return df


def _bucket_stats(df: pd.DataFrame) -> pd.DataFrame:
    if "tte_bucket" not in df.columns:
        # baseline-mode trades don't have bucket; derive from tte_s on the fly.
        from .taker_gate import _bucket_tte
        df = df.copy()
        df["tte_bucket"] = _bucket_tte(df["tte_s"])
    grp = df.groupby("tte_bucket", dropna=False)
    out = grp.agg(
        n=("net_pnl", "size"),
        total_net_pnl=("net_pnl", "sum"),
        net_pnl_mean=("net_pnl", "mean"),
        net_pnl_std=("net_pnl", "std"),
        hit_rate=("payoff", lambda s: float((s > df.loc[s.index, "fill_price"]).mean())),
    ).reset_index()
    out["t_stat"] = out["net_pnl_mean"] / (out["net_pnl_std"] / np.sqrt(out["n"]))
    return out


def _edge_deciles(df: pd.DataFrame) -> pd.DataFrame:
    if "abs_edge" not in df.columns:
        log.warning("no abs_edge column (baseline mode?); deriving from p_hybrid - market_prob")
        df = df.copy()
        df["abs_edge"] = (df["p_hybrid"] - df["market_prob"]).abs()
    try:
        df = df.copy()
        df["edge_decile"] = pd.qcut(df["abs_edge"], q=10, labels=False, duplicates="drop")
    except ValueError:
        df = df.copy()
        df["edge_decile"] = 0
    grp = df.groupby("edge_decile", dropna=False)
    out = grp.agg(
        n=("net_pnl", "size"),
        abs_edge_mean=("abs_edge", "mean"),
        total_net_pnl=("net_pnl", "sum"),
        net_pnl_mean=("net_pnl", "mean"),
    ).reset_index()
    return out


def _persistence_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("edge_persist_2s", "edge_persist_3s"):
        if col not in df.columns:
            continue
        for val, sub in df.groupby(col, dropna=False):
            rows.append({
                "persistence_column": col,
                "value": bool(val) if pd.notna(val) else None,
                "n": int(len(sub)),
                "total_net_pnl": float(sub["net_pnl"].sum()),
                "net_pnl_mean": float(sub["net_pnl"].mean()) if len(sub) else float("nan"),
            })
    return pd.DataFrame(rows)


def _plots(df: pd.DataFrame, bucket: pd.DataFrame) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed — skipping plots")
        return

    out_dir = derived("")
    # Trade count by bucket.
    fig, ax = plt.subplots()
    bucket.plot(kind="bar", x="tte_bucket", y="n", ax=ax, legend=False)
    ax.set_title("Taker trade count by tte bucket")
    ax.set_ylabel("n trades")
    fig.tight_layout()
    fig.savefig(out_dir / "trade_count_by_bucket.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    bucket.plot(kind="bar", x="tte_bucket", y="net_pnl_mean", ax=ax, legend=False)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Avg net PnL by tte bucket")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_pnl_by_bucket.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    sorted_df = df.sort_values("state_ts_ms").reset_index(drop=True)
    cum = sorted_df["net_pnl"].cumsum()
    ax.plot(cum.index, cum.values)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Cumulative net PnL (trade order)")
    ax.set_xlabel("trade index")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_pnl.png", dpi=120)
    plt.close(fig)


def run(make_plots: bool = True) -> dict:
    df = _load_trades()
    bucket = _bucket_stats(df)
    deciles = _edge_deciles(df)
    persistence = _persistence_stats(df)
    bucket.to_csv(derived("taker_gate_bucket_stats.csv"), index=False)
    deciles.to_csv(derived("taker_gate_edge_deciles.csv"), index=False)
    persistence.to_csv(derived("taker_gate_persistence_stats.csv"), index=False)
    log.info("bucket stats:\n%s", bucket.to_string(index=False))
    log.info("edge deciles:\n%s", deciles.to_string(index=False))
    if not persistence.empty:
        log.info("persistence:\n%s", persistence.to_string(index=False))
    if make_plots:
        _plots(df, bucket)
    return {
        "n_trades": int(len(df)),
        "by_bucket": bucket.to_dict(orient="records"),
        "by_edge_decile": deciles.to_dict(orient="records"),
        "by_persistence": persistence.to_dict(orient="records") if not persistence.empty else [],
    }


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-plots", dest="make_plots", action="store_false", default=True)
    args = ap.parse_args()
    run(make_plots=args.make_plots)


if __name__ == "__main__":
    _main()
