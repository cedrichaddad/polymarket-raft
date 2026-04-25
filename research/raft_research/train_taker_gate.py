"""Train a simple learned taker entry gate.

Target: a binary label per candidate row — `1` iff a trade entered at this row
would have yielded **strictly positive net PnL** after a conservative fill and
the configured fee assumption.

Model: standardized logistic regression (interpretable, cheap, robust on small
data). Persisted as a JSON artifact (feature_cols, scaler mean/scale, intercept,
coefs, metrics) — no pickled sklearn objects, so the scorer is side-effect free.

Evaluation modes:

* `--method global_time`: time-ordered fractional split by `state_ts_ms`.
* `--method grouped_market`: GroupKFold over `market_id` — same market never
  appears in both train and test within a fold. Reported metrics are the
  cross-validated mean.

Write:
    data/derived/taker_gate_model.json
    data/derived/taker_gate_model_metrics.json
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from .backtest_taker import _prepare_dataframe
from .paths import derived
from .taker_gate import GateConfig, add_gate_features

log = logging.getLogger(__name__)


MODEL_FEATURE_COLS = [
    "p_hybrid",
    "market_prob",
    "abs_edge",
    "tte_s",
    "spread_yes",
    "signed_flow_1s",
    "signed_flow_5s",
    "rv_10s",
    "rv_30s",
    "chainlink_binance_gap",
    "tick_regime",
    "edge_persist_2s",
    "edge_persist_3s",
    "mid_change_1s",
    "mid_change_2s",
]
BUCKET_DUMMY_COLS = ["bkt_15_30", "bkt_10_15", "bkt_5_10", "bkt_2_5", "bkt_0_2"]


def _simulate_net_pnl(row_df: pd.DataFrame, taker_fee_prob: float) -> np.ndarray:
    """Simulate net PnL for each row as if we entered on sign(edge_raw)."""
    side_is_buy = row_df["edge_raw"] > 0
    half_spread = (row_df["spread_yes"] / 2.0).clip(lower=0.0)
    fill_price = np.where(
        side_is_buy,
        row_df["market_prob"] + half_spread,
        row_df["market_prob"] - half_spread,
    ).clip(0.001, 0.999)
    payoff = np.where(side_is_buy, row_df["resolved_up"], 1 - row_df["resolved_up"])
    return payoff - fill_price - taker_fee_prob


def _build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a clean feature frame + list of column names."""
    X = df[MODEL_FEATURE_COLS].astype(float).copy()
    # Bool → float so the linear model can standardize uniformly.
    for col in ("edge_persist_2s", "edge_persist_3s"):
        X[col] = X[col].astype(float)
    # Bucket one-hots (5 columns, one per bucket label).
    for label in ("15_30", "10_15", "5_10", "2_5", "0_2"):
        X[f"bkt_{label}"] = (df["tte_bucket"] == label).astype(float)
    X = X.fillna(0.0)
    cols = MODEL_FEATURE_COLS + BUCKET_DUMMY_COLS
    return X[cols], cols


def _fit_one(X: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Standardize + fit. Returns (mean, scale, intercept, coefs)."""
    mean = X.mean(axis=0).to_numpy()
    scale = X.std(axis=0, ddof=0).replace(0.0, 1.0).to_numpy()
    Xs = (X.to_numpy() - mean) / scale
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xs, y)
    return mean, scale, float(clf.intercept_[0]), clf.coef_[0]


def _predict(X: pd.DataFrame, mean: np.ndarray, scale: np.ndarray,
             intercept: float, coefs: np.ndarray) -> np.ndarray:
    Xs = (X.to_numpy() - mean) / scale
    z = Xs @ coefs + intercept
    return 1.0 / (1.0 + np.exp(-z))


def train(
    method: str = "grouped_market",
    fraction: float = 0.7,
    taker_fee_prob: float = 0.0,
    sigma_per_sec: float = 5e-4,
    calibrator_path: str | None = None,
    max_entry_window_secs: float = 15.0,
    n_splits: int = 3,
    holdout_fraction: float | None = None,
) -> dict:
    cfg = GateConfig(max_entry_window_secs=max_entry_window_secs)
    # If a holdout is specified, train exclusively on dev-segment markets so the
    # model never sees holdout markets.
    df = _prepare_dataframe(
        sigma_per_sec, calibrator_path,
        holdout_fraction=holdout_fraction,
        holdout_segment=("dev" if holdout_fraction else None),
    )
    df = add_gate_features(df, cfg)

    # Candidate set for training: within entry window, acceptable quote quality.
    cand = df[(df["tte_s"] > 0) & (df["tte_s"] <= max_entry_window_secs)].copy()
    cand = cand[cand["quote_quality_flag"]].copy()
    if cand.empty:
        raise SystemExit("no candidate rows after quote-quality filter — nothing to train")

    cand["net_pnl_sim"] = _simulate_net_pnl(cand, taker_fee_prob)
    cand["y"] = (cand["net_pnl_sim"] > 0).astype(int)
    X, feature_cols = _build_feature_matrix(cand)
    y = cand["y"].to_numpy()

    metrics: dict = {
        "method": method,
        "n_candidates": int(len(cand)),
        "class_balance": float(y.mean()),
        "unique_markets": int(cand["market_id"].nunique()),
    }

    if method == "grouped_market":
        groups = cand["market_id"].to_numpy()
        n_groups = len(np.unique(groups))
        k = min(n_splits, n_groups)
        if k < 2:
            log.warning("grouped_market needs >=2 markets; falling back to global_time")
            method = "global_time"
        else:
            gkf = GroupKFold(n_splits=k)
            fold_metrics = []
            oof = np.full(len(X), np.nan)
            for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups)):
                if len(np.unique(y[tr_idx])) < 2:
                    log.warning("fold %d has single-class train; skipping", fold_idx)
                    continue
                mean, scale, b0, b = _fit_one(X.iloc[tr_idx], y[tr_idx])
                oof[te_idx] = _predict(X.iloc[te_idx], mean, scale, b0, b)
                fold_metrics.append(_score(y[te_idx], oof[te_idx]))
            metrics["folds"] = fold_metrics
            if fold_metrics:
                metrics["cv_brier"] = float(np.mean([f["brier"] for f in fold_metrics]))
                metrics["cv_logloss"] = float(np.mean([f["logloss"] for f in fold_metrics]))
                metrics["cv_auc"] = float(np.nanmean([f["auc"] for f in fold_metrics]))
            # Final model on full data for deployment.

    if method == "global_time":
        cand_sorted = cand.sort_values("state_ts_ms")
        order = cand_sorted.index
        X_sorted = X.loc[order]
        y_sorted = y[np.argsort(cand["state_ts_ms"].to_numpy())]
        n_tr = int(len(cand) * fraction)
        if n_tr < 10 or len(cand) - n_tr < 5:
            raise SystemExit(f"split too small: train={n_tr}, test={len(cand) - n_tr}")
        X_tr, X_te = X_sorted.iloc[:n_tr], X_sorted.iloc[n_tr:]
        y_tr, y_te = y_sorted[:n_tr], y_sorted[n_tr:]
        if len(np.unique(y_tr)) < 2:
            raise SystemExit("training set has a single class; cannot fit logistic")
        mean, scale, b0, b = _fit_one(X_tr, y_tr)
        test_scores = _predict(X_te, mean, scale, b0, b)
        metrics.update({"test": _score(y_te, test_scores)})
    else:
        # Fit a final model on the entire candidate set.
        mean, scale, b0, b = _fit_one(X, y)

    model = {
        "feature_cols": feature_cols,
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "intercept": float(b0),
        "coefs": b.tolist(),
        "taker_fee_prob": taker_fee_prob,
        "max_entry_window_secs": max_entry_window_secs,
    }
    out_model = derived("taker_gate_model.json")
    out_model.write_text(json.dumps(model, indent=2))
    out_metrics = derived("taker_gate_model_metrics.json")
    out_metrics.write_text(json.dumps(metrics, indent=2))
    log.info("wrote %s", out_model)
    log.info("metrics: %s", json.dumps(metrics, indent=2))
    return metrics


def _score(y: np.ndarray, p: np.ndarray) -> dict:
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p_clipped))
    return {
        "n": int(len(y)),
        "pos_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p_clipped)),
        "logloss": float(log_loss(y, p_clipped)),
        "auc": auc,
    }


def score_rows(df: pd.DataFrame, model_path: Path) -> np.ndarray:
    """Apply a persisted JSON model to a dataframe with gate features."""
    data = json.loads(Path(model_path).read_text())
    X, _ = _build_feature_matrix(df)
    X = X[data["feature_cols"]]
    mean = np.asarray(data["scaler_mean"])
    scale = np.asarray(data["scaler_scale"])
    return _predict(X, mean, scale, data["intercept"], np.asarray(data["coefs"]))


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["global_time", "grouped_market"], default="grouped_market")
    ap.add_argument("--fraction", type=float, default=0.7)
    ap.add_argument("--taker-fee-prob", type=float, default=0.0)
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--calibrator")
    ap.add_argument("--max-entry-window-secs", type=float, default=15.0)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--holdout-fraction", type=float, default=None,
                    help="Reserve fraction of newest markets for final holdout (not used in training)")
    args = ap.parse_args()
    train(
        method=args.method,
        fraction=args.fraction,
        taker_fee_prob=args.taker_fee_prob,
        sigma_per_sec=args.sigma_per_sec,
        calibrator_path=args.calibrator,
        max_entry_window_secs=args.max_entry_window_secs,
        n_splits=args.n_splits,
        holdout_fraction=args.holdout_fraction,
    )


if __name__ == "__main__":
    _main()
