"""Rolling-holdout sizing evaluation.

One 70/30 split can lie. This script runs K walk-forward market-level splits
and reports sizing metrics per split and averaged across splits.

Split scheme (walk-forward, market-level):
  1. Sort markets by their first `state_ts_ms`.
  2. Divide into `n_folds + 1` contiguous chunks of roughly equal market count.
  3. For fold i in [1 .. n_folds]:
       dev segment   = markets in chunks 0..i-1  (strictly older)
       holdout segment = markets in chunk i      (contiguous block)
     Fold 0 is skipped (no dev history).

For each fold and each sizing mode, we report the full metrics bundle.
Then we average across folds and print the distribution (mean, median, min).

Outputs:
  data/derived/taker_gate_rolling_sizing.csv      (per-fold rows)
  data/derived/taker_gate_rolling_sizing_summary.csv  (per-mode averages)

Rule gate is FROZEN (see TAKER_GATE_FINDINGS.md) — this script does not
tune entry selection. It tunes nothing, in fact; it reports stability.
"""
from __future__ import annotations
import argparse
import logging

import numpy as np
import pandas as pd

from .backtest_taker import _compute_pnl, _prepare_dataframe, _run_rule_gate
from .evaluate_sizing import FROZEN_RULE_GATE
from .paths import derived
from .sizing import SizingConfig, compute_sizes, metrics_bundle

log = logging.getLogger(__name__)


def _market_order(df: pd.DataFrame) -> list[str]:
    """Markets sorted by first state_ts_ms (oldest first)."""
    first = df.groupby("market_id")["state_ts_ms"].min().sort_values()
    return first.index.tolist()


def _split_markets(market_ids: list[str], n_folds: int) -> list[tuple[list[str], list[str]]]:
    """Return list of (dev_ids, holdout_ids) pairs for folds 1..n_folds."""
    chunks = np.array_split(market_ids, n_folds + 1)
    pairs: list[tuple[list[str], list[str]]] = []
    for i in range(1, len(chunks)):
        dev = np.concatenate(chunks[:i]).tolist()
        holdout = chunks[i].tolist()
        if dev and holdout:
            pairs.append((dev, holdout))
    return pairs


def _run_for_ids(df_full: pd.DataFrame, ids: list[str], taker_fee_prob: float) -> pd.DataFrame:
    sub = df_full[df_full["market_id"].isin(ids)].copy()
    entries = _run_rule_gate(sub, FROZEN_RULE_GATE, first_trigger_only=True)
    if entries.empty:
        return entries
    return _compute_pnl(entries, taker_fee_prob)


def _modes(fee: float, slip: float) -> dict[str, SizingConfig]:
    return {
        "flat": SizingConfig(mode="flat", fee=fee, slippage_haircut=slip),
        "bucketed": SizingConfig(mode="bucketed", fee=fee, slippage_haircut=slip),
        "fractional_kelly": SizingConfig(mode="fractional_kelly", fee=fee, slippage_haircut=slip),
    }


def run(
    sigma_per_sec: float = 5e-4,
    calibrator_path: str | None = None,
    taker_fee_prob: float = 0.006,
    slippage_haircut: float = 0.0,
    n_folds: int = 4,
) -> pd.DataFrame:
    df_full = _prepare_dataframe(sigma_per_sec, calibrator_path)
    ordered = _market_order(df_full)
    pairs = _split_markets(ordered, n_folds)
    log.info("rolling splits: %d folds, %d markets total", len(pairs), len(ordered))

    rows: list[dict] = []
    for fold_idx, (dev_ids, hold_ids) in enumerate(pairs, start=1):
        log.info("fold %d: dev=%d mkts, holdout=%d mkts", fold_idx, len(dev_ids), len(hold_ids))
        for segment_name, ids in (("dev", dev_ids), ("holdout", hold_ids)):
            trades = _run_for_ids(df_full, ids, taker_fee_prob)
            if trades.empty:
                log.warning("fold %d %s: no trades", fold_idx, segment_name)
                continue
            for mode_name, cfg in _modes(taker_fee_prob, slippage_haircut).items():
                sized = compute_sizes(trades, cfg)
                m = metrics_bundle(sized)
                m.update({
                    "fold": fold_idx,
                    "segment": segment_name,
                    "sizing_mode": mode_name,
                    "n_markets": len(ids),
                    "n_raw_trades": int(len(trades)),
                    "n_dropped_by_caps": int((sized["size"] == 0).sum()),
                })
                rows.append(m)

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        log.error("no rows produced"); return per_fold

    cols = ["fold", "segment", "sizing_mode", "n_markets", "n_raw_trades",
            "n_trades", "n_dropped_by_caps",
            "total", "mean", "t_stat", "hit_rate",
            "max_drawdown", "ret_over_dd", "worst_losing_streak"]
    per_fold = per_fold[[c for c in cols if c in per_fold.columns]]
    per_fold_p = derived("taker_gate_rolling_sizing.csv")
    per_fold.to_csv(per_fold_p, index=False)
    log.info("wrote %s", per_fold_p)

    # Aggregate across folds, per (segment, mode).
    num_cols = ["total", "mean", "t_stat", "hit_rate",
                "max_drawdown", "ret_over_dd", "worst_losing_streak",
                "n_trades", "n_raw_trades"]
    agg_rows: list[dict] = []
    for (seg, mode), sub in per_fold.groupby(["segment", "sizing_mode"]):
        out = {"segment": seg, "sizing_mode": mode, "n_folds": int(len(sub))}
        for c in num_cols:
            if c not in sub.columns: continue
            vals = pd.to_numeric(sub[c], errors="coerce").dropna().to_numpy()
            if len(vals) == 0:
                out[f"{c}_mean"] = None; out[f"{c}_median"] = None; out[f"{c}_min"] = None
            else:
                out[f"{c}_mean"] = float(np.mean(vals))
                out[f"{c}_median"] = float(np.median(vals))
                out[f"{c}_min"] = float(np.min(vals))
        # Stability: fraction of folds with positive total.
        pos = pd.to_numeric(sub["total"], errors="coerce").dropna() > 0
        out["frac_folds_positive"] = float(pos.mean()) if len(pos) else None
        agg_rows.append(out)

    summary = pd.DataFrame(agg_rows)
    summary_p = derived("taker_gate_rolling_sizing_summary.csv")
    summary.to_csv(summary_p, index=False)

    log.info("\n=== per-fold ===\n%s", per_fold.to_string(index=False))
    log.info("\n=== averaged ===\n%s", summary[[
        "segment", "sizing_mode", "n_folds",
        "total_mean", "total_median", "total_min",
        "t_stat_mean", "max_drawdown_mean", "ret_over_dd_mean",
        "frac_folds_positive",
    ]].to_string(index=False))
    log.info("wrote %s", summary_p)
    return summary


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--calibrator")
    ap.add_argument("--taker-fee-prob", type=float, default=0.006)
    ap.add_argument("--slippage-haircut", type=float, default=0.0)
    ap.add_argument("--n-folds", type=int, default=4)
    args = ap.parse_args()
    run(
        sigma_per_sec=args.sigma_per_sec,
        calibrator_path=args.calibrator,
        taker_fee_prob=args.taker_fee_prob,
        slippage_haircut=args.slippage_haircut,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    _main()
