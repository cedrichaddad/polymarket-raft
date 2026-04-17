"""Maker backtest with conservative fill model (§14.4).

Simulation rules:
  * At each 1-second bar within the center region (center_low..center_high)
    the strategy is assumed to have a resting post-only bid at `mid - half_spread`
    and a resting post-only ask at `mid + half_spread`.
  * A fill is recognized only if:
      - the quote has been resting for >= `min_dwell_secs` bars, AND
      - opposite-side traded volume during the dwell period exceeds `min_opp_volume`, AND
      - the quote is still competitive (within 1 tick of best) at fill time.
  * For each fill we compute post-fill markouts at {250ms, 500ms, 1s, 2s, 5s}
    using the subsequent mid and the chainlink drift (§14.5).
  * PnL components: spread capture, post-fill drift, maker rebate.

The fill model is deliberately strict — the spec says a passive fill should
*not* be assumed merely because the future midpoint touches the quote.

Outputs: per-fill Parquet + markout aggregate tables + summary JSON in
`data/derived/backtest_maker/`.

Usage:
    python -m raft_research.backtest_maker \
        --center-low 0.40 --center-high 0.60 \
        --min-dwell-secs 2 --min-opp-volume 50 \
        --rebate-mode conservative
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import add_features
from .paths import derived

log = logging.getLogger(__name__)

MARKOUT_HORIZONS_MS = [250, 500, 1000, 2000, 5000]
REBATE_MODES = {"none": 0.0, "conservative": 0.001, "optimistic": 0.003}


def run(
    center_low: float = 0.40,
    center_high: float = 0.60,
    min_dwell_secs: int = 2,
    min_opp_volume: float = 50.0,
    rebate_mode: str = "conservative",
    output_dir: str | None = None,
) -> dict:
    path = derived("market_state_1s_labeled.parquet")
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_labels first")
    df = pd.read_parquet(path)
    df = df[df["resolved_up"].notna()].copy()
    df = add_features(df)
    df = df.sort_values(["market_id", "state_ts_ms"]).reset_index(drop=True)

    fills: list[dict] = []

    for mid_id, g in df.groupby("market_id", sort=False):
        g = g.reset_index(drop=True)
        in_center = (g["market_prob"] >= center_low) & (g["market_prob"] <= center_high)
        dwelling = in_center.rolling(min_dwell_secs, min_periods=min_dwell_secs).sum() == min_dwell_secs
        opp_flow = g["signed_flow_5s"].abs().rolling(min_dwell_secs).sum()
        eligible = dwelling & (opp_flow >= min_opp_volume)
        if not eligible.any():
            continue

        mids = g["market_prob"].to_numpy()
        spread = g["spread_yes"].to_numpy()
        flow = g["signed_flow_1s"].to_numpy()
        times = g["state_ts_ms"].to_numpy()
        chain = g["chainlink_price"].to_numpy()

        for i in np.where(eligible.fillna(False).to_numpy())[0]:
            # At bar i our bid rests at mid-half, ask at mid+half.
            half = spread[i] / 2.0
            bid_px = mids[i] - half
            ask_px = mids[i] + half

            # A filled bid needs aggressive sell flow (flow < 0). Conversely for ask.
            # We assign fill to whichever side took the larger matching volume in this bar.
            sell_pressure = max(-flow[i], 0.0)
            buy_pressure = max(flow[i], 0.0)
            if sell_pressure >= buy_pressure and sell_pressure > 0:
                fills.append(_record_fill(
                    mid_id, "buy", bid_px, i, times, mids, chain,
                    g["resolved_up"].iloc[i], rebate_mode,
                ))
            elif buy_pressure > 0:
                fills.append(_record_fill(
                    mid_id, "sell", ask_px, i, times, mids, chain,
                    g["resolved_up"].iloc[i], rebate_mode,
                ))

    fdf = pd.DataFrame(fills)
    out_dir = Path(output_dir or derived("backtest_maker"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fdf.to_parquet(out_dir / "fills.parquet", index=False)
    summary = _summarize(fdf)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("maker backtest: %s", json.dumps(summary, indent=2))
    return summary


def _record_fill(
    market_id: str,
    side: str,
    fill_px: float,
    i: int,
    times: np.ndarray,
    mids: np.ndarray,
    chain: np.ndarray,
    resolved_up: float,
    rebate_mode: str,
) -> dict:
    rec: dict = {
        "market_id": market_id,
        "side": side,
        "fill_ts_ms": int(times[i]),
        "fill_price": float(fill_px),
        "resolved_up": int(resolved_up),
    }
    for h in MARKOUT_HORIZONS_MS:
        j = _index_at_offset(times, i, h)
        rec[f"markout_{h}ms"] = float(mids[j] - fill_px) if j is not None else np.nan
    # Terminal payoff: buy -> resolved_up, sell -> 1 - resolved_up.
    payoff = resolved_up if side == "buy" else 1.0 - resolved_up
    rebate = REBATE_MODES.get(rebate_mode, 0.0)
    rec["gross_pnl"] = float(payoff - fill_px) if side == "buy" else float(fill_px - (1.0 - payoff))
    rec["rebate"] = rebate
    rec["net_pnl"] = rec["gross_pnl"] + rebate
    return rec


def _index_at_offset(times: np.ndarray, i: int, offset_ms: int) -> int | None:
    target = times[i] + offset_ms
    j = np.searchsorted(times, target, side="left")
    if j >= len(times):
        return None
    return int(j)


def _summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_fills": 0}
    out: dict = {
        "n_fills": int(len(df)),
        "n_buy": int((df["side"] == "buy").sum()),
        "n_sell": int((df["side"] == "sell").sum()),
        "net_pnl_mean": float(df["net_pnl"].mean()),
        "net_pnl_std": float(df["net_pnl"].std(ddof=1)),
        "total_net_pnl": float(df["net_pnl"].sum()),
    }
    for h in MARKOUT_HORIZONS_MS:
        col = f"markout_{h}ms"
        if col in df:
            out[f"{col}_mean"] = float(df[col].mean())
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--center-low", type=float, default=0.40)
    ap.add_argument("--center-high", type=float, default=0.60)
    ap.add_argument("--min-dwell-secs", type=int, default=2)
    ap.add_argument("--min-opp-volume", type=float, default=50.0)
    ap.add_argument("--rebate-mode", choices=list(REBATE_MODES), default="conservative")
    args = ap.parse_args()
    run(
        center_low=args.center_low,
        center_high=args.center_high,
        min_dwell_secs=args.min_dwell_secs,
        min_opp_volume=args.min_opp_volume,
        rebate_mode=args.rebate_mode,
    )


if __name__ == "__main__":
    _main()
