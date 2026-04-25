"""Evaluate sizing modes on top of the frozen rule_gate.

Runs the rule_gate taker backtest on dev and holdout segments, then applies
three sizing modes (flat, bucketed, fractional_kelly) to each. Reports a
bundle of metrics per (segment, sizing_mode) and writes
``data/derived/taker_gate_sizing_eval.csv``.

Principles baked in:
  * rule_gate config is the *frozen* default (see TAKER_GATE_FINDINGS.md).
  * Sizing is tuned on dev and reported on holdout.
  * No tuning is done inside this script — it just reports. Tuning should
    happen in a separate sweep on dev and the winning config then locked in.

Usage:
    python -m raft_research.evaluate_sizing \\
        --calibrator ../data/derived/calibrator.json \\
        --holdout-fraction 0.3
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest_taker import _compute_pnl, _prepare_dataframe, _run_rule_gate
from .paths import derived
from .sizing import SizingConfig, compute_sizes, metrics_bundle
from .taker_gate import GateConfig

log = logging.getLogger(__name__)


FROZEN_RULE_GATE = GateConfig(
    max_entry_window_secs=15.0,
    persist_bars=2,
    require_persistence=True,
    thresholds={"15_30": None, "10_15": 0.03, "5_10": 0.025, "2_5": 0.02, "0_2": None},
    max_spread=0.05,
    max_chainlink_binance_gap_abs=500.0,
)


def _run_segment(
    segment: str | None,
    sigma_per_sec: float,
    calibrator_path: str | None,
    holdout_fraction: float | None,
    taker_fee_prob: float,
) -> pd.DataFrame:
    """Run rule_gate on one segment (dev/holdout/None=full) and return trades."""
    df = _prepare_dataframe(
        sigma_per_sec, calibrator_path,
        holdout_fraction=holdout_fraction,
        holdout_segment=segment,
    )
    entries = _run_rule_gate(df, FROZEN_RULE_GATE, first_trigger_only=True)
    if entries.empty:
        return entries
    entries = _compute_pnl(entries, taker_fee_prob)
    # end_ts_ms comes from the labeled state; preserve it for sizing caps.
    return entries


def _sizing_configs(fee: float, slip: float) -> dict[str, SizingConfig]:
    return {
        "flat": SizingConfig(mode="flat", fee=fee, slippage_haircut=slip),
        "bucketed": SizingConfig(mode="bucketed", fee=fee, slippage_haircut=slip),
        "fractional_kelly": SizingConfig(mode="fractional_kelly", fee=fee, slippage_haircut=slip),
    }


def run(
    sigma_per_sec: float = 5e-4,
    calibrator_path: str | None = None,
    holdout_fraction: float = 0.3,
    taker_fee_prob: float = 0.006,
    slippage_haircut: float = 0.0,
    output_csv: str | None = None,
) -> pd.DataFrame:
    rows = []
    per_trade_frames = []
    for segment in ("dev", "holdout"):
        trades = _run_segment(segment, sigma_per_sec, calibrator_path,
                              holdout_fraction, taker_fee_prob)
        if trades.empty:
            log.warning("no trades in segment %s", segment)
            continue
        for name, cfg in _sizing_configs(taker_fee_prob, slippage_haircut).items():
            sized = compute_sizes(trades, cfg)
            m = metrics_bundle(sized)
            m["segment"] = segment
            m["sizing_mode"] = name
            m["n_dropped_by_caps"] = int((sized["size"] == 0).sum())
            rows.append(m)

            tag = sized.assign(segment=segment, sizing_mode=name)
            per_trade_frames.append(tag)

    out = pd.DataFrame(rows)
    if out.empty:
        log.error("no rows produced — check data + calibrator path")
        return out

    # Re-order columns for readability.
    cols = ["segment", "sizing_mode", "n_trades", "n_active",
            "total", "mean", "std", "t_stat", "hit_rate",
            "max_drawdown", "ret_over_dd", "worst_losing_streak",
            "n_dropped_by_caps"]
    out = out[[c for c in cols if c in out.columns]]

    out_p = Path(output_csv) if output_csv else derived("taker_gate_sizing_eval.csv")
    out.to_csv(out_p, index=False)
    log.info("wrote %s\n%s", out_p, out.to_string(index=False))

    if per_trade_frames:
        trades_out = derived("taker_gate_sizing_trades.parquet")
        pd.concat(per_trade_frames, ignore_index=True).to_parquet(trades_out)
        log.info("wrote %s", trades_out)
    return out


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--calibrator")
    ap.add_argument("--holdout-fraction", type=float, default=0.3)
    ap.add_argument("--taker-fee-prob", type=float, default=0.006)
    ap.add_argument("--slippage-haircut", type=float, default=0.0)
    ap.add_argument("--output-csv")
    args = ap.parse_args()
    run(
        sigma_per_sec=args.sigma_per_sec,
        calibrator_path=args.calibrator,
        holdout_fraction=args.holdout_fraction,
        taker_fee_prob=args.taker_fee_prob,
        slippage_haircut=args.slippage_haircut,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    _main()
