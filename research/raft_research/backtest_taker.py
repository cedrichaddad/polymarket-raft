"""Taker backtest (§14.6).

For each labeled state row inside the taker window we simulate:
    edge = |p_hybrid - p_market| - all_in_threshold
    if edge > 0:
        take on the side p_hybrid > p_market predicts
        realized PnL per unit notional = (payoff - fill_price) - fee

Where:
    payoff = 1 if resolved_up and we took YES (buy), else 0
             1 - resolved_up if we took NO (sell)
    fill_price = market_prob + half-spread (conservative)
    fee = `taker_fee_prob` in probability units

Outputs a per-trade Parquet + aggregate JSON to `data/derived/backtest_taker/`.

Usage:
    python -m raft_research.backtest_taker --taker-window-secs 30 \
        --taker-fee-prob 0.0 --all-in 0.012
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fair_value import fair_prob
from .features import add_features
from .paths import derived

log = logging.getLogger(__name__)


def run(
    taker_window_secs: float = 30.0,
    taker_fee_prob: float = 0.0,
    all_in_threshold: float = 0.012,
    sigma_per_sec: float = 5e-4,
    output_dir: str | None = None,
    calibrator_path: str | None = None,
) -> dict:
    path = derived("market_state_1s_labeled.parquet")
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_labels first")
    df = pd.read_parquet(path)
    df = df[df["resolved_up"].notna()].copy()
    df = add_features(df)

    df["tte_s"] = df["tte_ms"] / 1000.0
    df["p_0"] = fair_prob(df["chainlink_price"], df["open_ref_price"], df["tte_s"], sigma_per_sec)

    if calibrator_path:
        p_star = _apply_isotonic(df["p_0"], Path(calibrator_path))
        df["p_star"] = p_star
    else:
        df["p_star"] = df["p_0"]
    # For v1 we treat p_hybrid = p_star (retrieval overlay tested separately).
    df["p_hybrid"] = df["p_star"]

    within_taker = (df["tte_s"] > 0) & (df["tte_s"] <= taker_window_secs)
    df = df[within_taker].copy()

    edge = df["p_hybrid"] - df["market_prob"]
    df["side_is_buy"] = edge > 0
    df["abs_edge"] = edge.abs()
    df = df[df["abs_edge"] > all_in_threshold].copy()
    if df.empty:
        log.warning("no trades triggered at threshold=%s", all_in_threshold)

    # Conservative fill price: cross the full half-spread.
    half_spread = (df["spread_yes"] / 2.0).clip(lower=0.0)
    df["fill_price"] = np.where(
        df["side_is_buy"], df["market_prob"] + half_spread, df["market_prob"] - half_spread
    ).clip(0.001, 0.999)

    # Payoff: 1 if we bought YES and it resolved up, else 0. If we sold YES we get
    # 1 - resolved_up.
    payoff_buy = df["resolved_up"]
    payoff_sell = 1 - df["resolved_up"]
    df["payoff"] = np.where(df["side_is_buy"], payoff_buy, payoff_sell)
    df["gross_pnl"] = df["payoff"] - df["fill_price"]
    df["fee"] = taker_fee_prob
    df["net_pnl"] = df["gross_pnl"] - df["fee"]

    out_dir = Path(output_dir or derived("backtest_taker"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "trades.parquet", index=False)

    summary = _summarize(df)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("taker backtest: %s", json.dumps(summary, indent=2))
    return summary


def _summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0}
    mean = df["net_pnl"].mean()
    std = df["net_pnl"].std(ddof=1)
    t_stat = mean / (std / np.sqrt(len(df))) if std > 0 else float("nan")
    return {
        "n_trades": int(len(df)),
        "net_pnl_mean": float(mean),
        "net_pnl_std": float(std),
        "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "hit_rate": float((df["payoff"] > df["fill_price"]).mean()),
        "total_net_pnl": float(df["net_pnl"].sum()),
    }


def _apply_isotonic(p0: pd.Series, path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    bp = np.asarray(data["isotonic_breakpoints"])
    xs = bp[:, 0]
    ys = bp[:, 1]
    return np.interp(p0.to_numpy(), xs, ys)


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--taker-window-secs", type=float, default=30.0)
    ap.add_argument("--taker-fee-prob", type=float, default=0.0)
    ap.add_argument("--all-in", type=float, default=0.012)
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--calibrator")
    args = ap.parse_args()
    run(
        taker_window_secs=args.taker_window_secs,
        taker_fee_prob=args.taker_fee_prob,
        all_in_threshold=args.all_in,
        sigma_per_sec=args.sigma_per_sec,
        calibrator_path=args.calibrator,
    )


if __name__ == "__main__":
    _main()
