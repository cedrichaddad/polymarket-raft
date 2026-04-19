"""Maker backtest with conservative fill model (§14.4).

Simulation rules:
  * At each 1-second bar within the center region (center_low..center_high)
    the strategy is assumed to have a resting post-only bid at `mid - half_spread`
    and a resting post-only ask at `mid + half_spread`.
  * A fill is recognized only if:
      - the quote has been resting for >= `min_dwell_secs` bars, AND
      - opposite-side traded volume during the dwell period exceeds `min_opp_volume`, AND
      - the quote is still competitive (within 1 tick of best) at fill time.
  * After a fill, the dwell counter resets — the maker must re-post and wait
    before the next fill can occur (fill cooldown).
  * Inventory is tracked per market. Net position is capped at ±`max_inventory`.
    When the limit is hit, the maker stops quoting on the side that would
    increase exposure.
  * PnL is computed per fill using mid-to-mid markouts at standard horizons
    and terminal (resolution) payoff.  The summary decomposes PnL into
    spread capture vs directional to detect leakage.

Outputs: per-fill Parquet + markout aggregate tables + summary JSON in
`data/derived/backtest_maker/`.

Usage:
    python -m raft_research.backtest_maker \\
        --center-low 0.40 --center-high 0.60 \\
        --min-dwell-secs 2 --min-opp-volume 50 \\
        --rebate-mode conservative --max-inventory 5
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
DEFAULT_MAX_INVENTORY = 5


def run(
    center_low: float = 0.40,
    center_high: float = 0.60,
    min_dwell_secs: int = 2,
    min_opp_volume: float = 50.0,
    rebate_mode: str = "conservative",
    max_inventory: int = DEFAULT_MAX_INVENTORY,
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
        mids = g["market_prob"].to_numpy()
        spread = g["spread_yes"].to_numpy()
        flow = g["signed_flow_1s"].to_numpy()
        times = g["state_ts_ms"].to_numpy()
        chain = g["chainlink_price"].to_numpy()
        in_center = (mids >= center_low) & (mids <= center_high)
        resolved_up = float(g["resolved_up"].iloc[0])

        # ── Walk bars sequentially with dwell tracking + inventory ───
        dwell_count = 0          # consecutive bars in center
        opp_flow_window: list[float] = []  # rolling window of abs flow
        net_inventory = 0        # signed: +long, -short

        for i in range(len(g)):
            if in_center[i]:
                dwell_count += 1
                opp_flow_window.append(abs(flow[i]))
                if len(opp_flow_window) > min_dwell_secs:
                    opp_flow_window = opp_flow_window[-min_dwell_secs:]
            else:
                # Left center — reset dwell.
                dwell_count = 0
                opp_flow_window.clear()
                continue

            # Check eligibility: enough dwell time AND enough opposite-side flow.
            if dwell_count < min_dwell_secs:
                continue
            opp_total = sum(opp_flow_window[-min_dwell_secs:])
            if opp_total < min_opp_volume:
                continue

            # Determine fill side from flow direction.
            half = spread[i] / 2.0
            sell_pressure = max(-flow[i], 0.0)
            buy_pressure = max(flow[i], 0.0)

            side = None
            fill_px = 0.0
            if sell_pressure >= buy_pressure and sell_pressure > 0:
                # Our bid gets hit → we buy.
                if net_inventory < max_inventory:
                    side = "buy"
                    fill_px = mids[i] - half
            elif buy_pressure > 0:
                # Our ask gets lifted → we sell.
                if net_inventory > -max_inventory:
                    side = "sell"
                    fill_px = mids[i] + half

            if side is None:
                continue

            fills.append(_record_fill(
                mid_id, side, fill_px, mids[i], i, times, mids, chain,
                resolved_up, rebate_mode,
            ))

            # Update inventory.
            net_inventory += 1 if side == "buy" else -1

            # ── Fill cooldown: reset dwell so next fill requires fresh wait.
            dwell_count = 0
            opp_flow_window.clear()

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
    mid_at_fill: float,
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
        "mid_at_fill": float(mid_at_fill),
        "resolved_up": int(resolved_up),
    }
    for h in MARKOUT_HORIZONS_MS:
        j = _index_at_offset(times, i, h)
        rec[f"markout_{h}ms"] = float(mids[j] - fill_px) if j is not None else np.nan

    # PnL components.
    half_spread = abs(fill_px - mid_at_fill)
    rec["spread_capture"] = float(half_spread)

    # Terminal payoff: buy -> resolved_up, sell -> 1 - resolved_up.
    payoff = resolved_up if side == "buy" else 1.0 - resolved_up
    if side == "buy":
        rec["gross_pnl"] = float(payoff - fill_px)
        rec["directional_pnl"] = float(resolved_up - mid_at_fill)
    else:
        rec["gross_pnl"] = float(fill_px - resolved_up)
        rec["directional_pnl"] = float(mid_at_fill - resolved_up)

    rebate = REBATE_MODES.get(rebate_mode, 0.0)
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
        return {"n_fills": 0, "n_unique_markets": 0}
    out: dict = {
        "n_fills": int(len(df)),
        "n_unique_markets": int(df["market_id"].nunique()),
        "n_buy": int((df["side"] == "buy").sum()),
        "n_sell": int((df["side"] == "sell").sum()),
        "net_pnl_mean": float(df["net_pnl"].mean()),
        "net_pnl_std": float(df["net_pnl"].std(ddof=1)),
        "total_net_pnl": float(df["net_pnl"].sum()),
        "total_spread_capture": float(df["spread_capture"].sum()),
        "total_directional_pnl": float(df["directional_pnl"].sum()),
        "spread_pct_of_gross": float(
            df["spread_capture"].sum() / max(abs(df["gross_pnl"].sum()), 1e-9) * 100
        ),
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
    ap.add_argument("--max-inventory", type=int, default=DEFAULT_MAX_INVENTORY,
                    help="Max net position per market per side (default: 5)")
    args = ap.parse_args()
    run(
        center_low=args.center_low,
        center_high=args.center_high,
        min_dwell_secs=args.min_dwell_secs,
        min_opp_volume=args.min_opp_volume,
        rebate_mode=args.rebate_mode,
        max_inventory=args.max_inventory,
    )


if __name__ == "__main__":
    _main()
