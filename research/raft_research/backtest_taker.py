"""Taker backtest (§14.6).

Three modes, selected via `--mode`:

* `baseline` (default, original behavior):
    edge = |p_hybrid - p_market| - all_in_threshold
    if edge > 0 within `taker_window_secs`:
        take on the side p_hybrid > p_market predicts
    — One trade per market at first edge trigger.

* `rule_gate`:
    Apply `taker_gate.select_taker_entries` with a `GateConfig`. Quote-quality,
    per-bucket threshold, persistence all enforced. Still first-trigger-per-market
    by default.

* `model_gate`:
    Same entry window as rule_gate, but accept a row iff a learned classifier
    (from `train_taker_gate.py`) predicts positive net PnL with probability
    above `--model-gate-threshold`.

All modes write to `data/derived/backtest_taker/` unless `--output-dir` is given.
They share the same fill-price, payoff, and summary machinery so results are
comparable.

Trade deduplication: Only the **first** 1-second bar per market where the gate
accepts counts as a trade entry. Subsequent bars in the same market are
autocorrelated snapshots of the same position, not independent bets.

Outputs: per-trade Parquet + aggregate JSON under `data/derived/backtest_taker/`.

Usage:
    python -m raft_research.backtest_taker                   # baseline
    python -m raft_research.backtest_taker --mode rule_gate
    python -m raft_research.backtest_taker --mode model_gate \\
        --gate-model ../data/derived/taker_gate_model.json
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .fair_value import fair_prob
from .features import FEATURE_COLS, add_features
from .paths import derived
from .taker_gate import GateConfig, add_gate_features, select_taker_entries

log = logging.getLogger(__name__)


# ─── core helpers ────────────────────────────────────────────────────────────

def _prepare_dataframe(
    sigma_per_sec: float,
    calibrator_path: str | None,
    holdout_fraction: float | None = None,
    holdout_segment: str | None = None,
) -> pd.DataFrame:
    """Load labeled state, attach features, `p_0`, `p_star`, and `p_hybrid`.

    If `holdout_fraction` is set (e.g. 0.3), markets are sorted by their first
    `state_ts_ms` and only the newest `holdout_fraction * n_markets` markets are
    returned (if `holdout_segment='holdout'`) or only the oldest `1 - fraction`
    markets (if `holdout_segment='dev'`). Full dataset when `holdout_segment`
    is None.
    """
    path = derived("market_state_1s_labeled.parquet")
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_labels first")
    df = pd.read_parquet(path)
    df = df[df["resolved_up"].notna()].copy()

    if holdout_fraction is not None and holdout_segment is not None:
        first_ts = df.groupby("market_id")["state_ts_ms"].min().sort_values()
        n_markets = len(first_ts)
        n_holdout = max(1, int(n_markets * holdout_fraction))
        if holdout_segment == "holdout":
            keep = first_ts.index[-n_holdout:]
        elif holdout_segment == "dev":
            keep = first_ts.index[:n_markets - n_holdout]
        else:
            raise SystemExit(f"unknown holdout_segment: {holdout_segment}")
        df = df[df["market_id"].isin(keep)].copy()

    df = add_features(df)

    df["tte_s"] = df["tte_ms"] / 1000.0
    df["p_0"] = fair_prob(df["chainlink_price"], df["open_ref_price"], df["tte_s"], sigma_per_sec)

    if calibrator_path and Path(calibrator_path).exists():
        df["p_star"] = _apply_calibrator(df["p_0"], df["market_id"], Path(calibrator_path))
    else:
        df["p_star"] = df["p_0"]
    # v1: p_hybrid = p_star. The retrieval overlay is evaluated separately in
    # compare.py; here we keep the taker signal interpretable.
    df["p_hybrid"] = df["p_star"]
    return df


def _compute_pnl(df: pd.DataFrame, taker_fee_prob: float) -> pd.DataFrame:
    """Attach fill_price, payoff, gross_pnl, net_pnl columns. Mutates a copy."""
    df = df.copy()
    # Conservative fill price: cross the full half-spread.
    half_spread = (df["spread_yes"] / 2.0).clip(lower=0.0)
    df["fill_price"] = np.where(
        df["side_is_buy"], df["market_prob"] + half_spread, df["market_prob"] - half_spread
    ).clip(0.001, 0.999)
    payoff_buy = df["resolved_up"]
    payoff_sell = 1 - df["resolved_up"]
    df["payoff"] = np.where(df["side_is_buy"], payoff_buy, payoff_sell)
    df["gross_pnl"] = df["payoff"] - df["fill_price"]
    df["fee"] = taker_fee_prob
    df["net_pnl"] = df["gross_pnl"] - df["fee"]
    return df


def _summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0, "n_unique_markets": 0}
    mean = df["net_pnl"].mean()
    std = df["net_pnl"].std(ddof=1) if len(df) > 1 else 0.0
    n = len(df)
    t_stat = mean / (std / np.sqrt(n)) if std and std > 0 else float("nan")
    summary = {
        "n_trades": int(n),
        "n_unique_markets": int(df["market_id"].nunique()),
        "net_pnl_mean": float(mean),
        "net_pnl_std": float(std),
        "t_stat": float(t_stat) if not np.isnan(t_stat) else None,
        "hit_rate": float((df["payoff"] > df["fill_price"]).mean()),
        "total_net_pnl": float(df["net_pnl"].sum()),
    }
    # Per-bucket breakdown when gate-features are attached.
    if "tte_bucket" in df.columns:
        by_bucket = (
            df.groupby("tte_bucket", dropna=False)
              .agg(n=("net_pnl", "size"), total_net_pnl=("net_pnl", "sum"),
                   net_pnl_mean=("net_pnl", "mean"))
              .reset_index()
        )
        summary["by_tte_bucket"] = by_bucket.to_dict(orient="records")
    return summary


def _apply_calibrator(p0: pd.Series, market_ids: pd.Series, path: Path) -> np.ndarray:
    """Apply the calibrator, respecting leave-one-market-out folds if present."""
    data = json.loads(path.read_text())
    if "lomo_folds" in data:
        out = np.full(len(p0), np.nan, dtype=float)
        for mid, fold in data["lomo_folds"].items():
            mask = market_ids.to_numpy() == mid
            if not mask.any():
                continue
            bp = np.asarray(fold["isotonic_breakpoints"])
            if len(bp) == 0:
                out[mask] = p0.to_numpy()[mask]
            else:
                out[mask] = np.interp(p0.to_numpy()[mask], bp[:, 0], bp[:, 1])
        remaining = np.isnan(out)
        out[remaining] = p0.to_numpy()[remaining]
        return out
    bp = np.asarray(data["isotonic_breakpoints"])
    return np.interp(p0.to_numpy(), bp[:, 0], bp[:, 1])


# ─── mode implementations ───────────────────────────────────────────────────

def _run_baseline(
    df: pd.DataFrame,
    taker_window_secs: float,
    all_in_threshold: float,
    first_trigger_only: bool,
) -> pd.DataFrame:
    within = (df["tte_s"] > 0) & (df["tte_s"] <= taker_window_secs)
    cand = df[within].copy()
    edge = cand["p_hybrid"] - cand["market_prob"]
    cand["side_is_buy"] = edge > 0
    cand["abs_edge"] = edge.abs()
    cand = cand[cand["abs_edge"] > all_in_threshold].copy()
    if first_trigger_only and not cand.empty:
        cand = cand.sort_values(["market_id", "state_ts_ms"])
        cand = cand.groupby("market_id", sort=False).first().reset_index()
    return cand


def _run_rule_gate(
    df: pd.DataFrame,
    gate_config: GateConfig,
    first_trigger_only: bool,
) -> pd.DataFrame:
    df = add_gate_features(df, gate_config)
    entries = select_taker_entries(df, gate_config, first_trigger_only=first_trigger_only)
    if not entries.empty:
        entries["side_is_buy"] = entries["edge_raw"] > 0
    return entries


def _run_model_gate(
    df: pd.DataFrame,
    gate_config: GateConfig,
    gate_model_path: Path,
    model_threshold: float,
    first_trigger_only: bool,
) -> pd.DataFrame:
    # Reuse the rule-gate entry window + quote-quality + features, but replace
    # threshold/persistence rules with the learned classifier's score.
    df = add_gate_features(df, gate_config)
    cand = df[(df["tte_s"] > 0) & (df["tte_s"] <= gate_config.max_entry_window_secs)].copy()
    cand = cand[cand["quote_quality_flag"]].copy()
    if cand.empty:
        return cand

    from .train_taker_gate import score_rows  # local import to avoid sklearn at import time
    cand["gate_score"] = score_rows(cand, gate_model_path)
    cand["side_is_buy"] = cand["edge_raw"] > 0
    cand = cand[cand["gate_score"] >= model_threshold].copy()

    if first_trigger_only and not cand.empty:
        cand = cand.sort_values(["market_id", "state_ts_ms"])
        cand = cand.groupby("market_id", sort=False).first().reset_index()
    return cand


# ─── public entry point ─────────────────────────────────────────────────────

def run(
    mode: str = "baseline",
    taker_window_secs: float = 30.0,
    taker_fee_prob: float = 0.0,
    all_in_threshold: float = 0.012,
    sigma_per_sec: float = 5e-4,
    output_dir: str | None = None,
    calibrator_path: str | None = None,
    gate_config: GateConfig | None = None,
    gate_model_path: str | None = None,
    model_gate_threshold: float = 0.5,
    first_trigger_only: bool = True,
    holdout_fraction: float | None = None,
    holdout_segment: str | None = None,
) -> dict:
    df = _prepare_dataframe(sigma_per_sec, calibrator_path, holdout_fraction, holdout_segment)

    if mode == "baseline":
        entries = _run_baseline(df, taker_window_secs, all_in_threshold, first_trigger_only)
    elif mode == "rule_gate":
        cfg = gate_config or GateConfig()
        entries = _run_rule_gate(df, cfg, first_trigger_only)
    elif mode == "model_gate":
        if not gate_model_path or not Path(gate_model_path).exists():
            raise SystemExit(f"--gate-model file required for model_gate mode: {gate_model_path}")
        cfg = gate_config or GateConfig()
        entries = _run_model_gate(df, cfg, Path(gate_model_path), model_gate_threshold, first_trigger_only)
    else:
        raise SystemExit(f"unknown mode: {mode}")

    if entries.empty:
        log.warning("no trades triggered (mode=%s)", mode)
        # Still write an empty trades + summary so downstream tools don't 500.
        entries = entries.assign(
            side_is_buy=pd.Series(dtype=bool),
            fill_price=pd.Series(dtype=float),
            payoff=pd.Series(dtype=float),
            gross_pnl=pd.Series(dtype=float),
            fee=pd.Series(dtype=float),
            net_pnl=pd.Series(dtype=float),
        )
    else:
        entries = _compute_pnl(entries, taker_fee_prob)

    out_dir = Path(output_dir or derived("backtest_taker"))
    out_dir.mkdir(parents=True, exist_ok=True)
    entries.to_parquet(out_dir / "trades.parquet", index=False)

    summary = _summarize(entries)
    summary["mode"] = mode
    summary["params"] = {
        "taker_window_secs": taker_window_secs,
        "taker_fee_prob": taker_fee_prob,
        "all_in_threshold": all_in_threshold,
        "sigma_per_sec": sigma_per_sec,
        "model_gate_threshold": model_gate_threshold,
        "first_trigger_only": first_trigger_only,
        "gate_config": gate_config.to_dict() if gate_config else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("taker backtest: %s", json.dumps(
        {k: v for k, v in summary.items() if k != "by_tte_bucket"}, indent=2))
    return summary


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "rule_gate", "model_gate"], default="baseline")
    ap.add_argument("--taker-window-secs", type=float, default=30.0,
                    help="baseline mode: entry window. (rule/model_gate use --max-entry-window-secs.)")
    ap.add_argument("--taker-fee-prob", type=float, default=0.0)
    ap.add_argument("--all-in", type=float, default=0.012,
                    help="baseline mode: min |edge| to enter.")
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--calibrator")
    ap.add_argument("--output-dir")
    # Gate-mode knobs.
    ap.add_argument("--gate-config", help="path to JSON GateConfig override")
    ap.add_argument("--gate-model", help="path to JSON model file (for model_gate)")
    ap.add_argument("--max-entry-window-secs", type=float, default=15.0)
    ap.add_argument("--persist-bars", type=int, default=2)
    ap.add_argument("--require-persistence", dest="require_persistence", action="store_true", default=True)
    ap.add_argument("--no-persistence", dest="require_persistence", action="store_false")
    ap.add_argument("--first-trigger-only", dest="first_trigger_only", action="store_true", default=True)
    ap.add_argument("--all-bars", dest="first_trigger_only", action="store_false",
                    help="Experimental: count every accepted bar as a trade (NOT default).")
    ap.add_argument("--model-gate-threshold", type=float, default=0.5)
    ap.add_argument("--holdout-fraction", type=float, default=None,
                    help="Fraction of newest markets to reserve for holdout (e.g. 0.3)")
    ap.add_argument("--holdout-segment", choices=["dev", "holdout"], default=None,
                    help="Which side of the market-level split to evaluate on")
    args = ap.parse_args()

    # Build the gate config for the non-baseline modes.
    if args.mode in ("rule_gate", "model_gate"):
        if args.gate_config and Path(args.gate_config).exists():
            cfg = GateConfig.from_dict(json.loads(Path(args.gate_config).read_text()))
        else:
            cfg = GateConfig(
                max_entry_window_secs=args.max_entry_window_secs,
                persist_bars=args.persist_bars,
                require_persistence=args.require_persistence,
            )
    else:
        cfg = None

    run(
        mode=args.mode,
        taker_window_secs=args.taker_window_secs,
        taker_fee_prob=args.taker_fee_prob,
        all_in_threshold=args.all_in,
        sigma_per_sec=args.sigma_per_sec,
        calibrator_path=args.calibrator,
        output_dir=args.output_dir,
        gate_config=cfg,
        gate_model_path=args.gate_model,
        model_gate_threshold=args.model_gate_threshold,
        first_trigger_only=args.first_trigger_only,
        holdout_fraction=args.holdout_fraction,
        holdout_segment=args.holdout_segment,
    )


if __name__ == "__main__":
    _main()
