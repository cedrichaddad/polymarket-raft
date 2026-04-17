"""Fit the p_0 -> p* calibrator (§11.2).

Trains an isotonic regression on a time-ordered split and writes the
calibrator + metrics to `data/derived/calibrator.json`. Optionally fits a
logistic regression on a small hand-crafted feature set as a baseline.

Usage:
    python -m raft_research.calibrate --train-until 2026-04-01 --test-from 2026-04-01
"""
from __future__ import annotations
import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .fair_value import fair_prob
from .features import FEATURE_COLS, add_features
from .paths import derived

log = logging.getLogger(__name__)


def _load_labeled() -> pd.DataFrame:
    path = derived("market_state_1s_labeled.parquet")
    if not path.exists():
        raise SystemExit(f"missing {path} — run build_labels first")
    return pd.read_parquet(path)


def _split_by_date(df: pd.DataFrame, train_until: str, test_from: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    t_until = pd.Timestamp(train_until).tz_localize("UTC").value // 10**6
    t_from = pd.Timestamp(test_from).tz_localize("UTC").value // 10**6
    tr = df[df["state_ts_ms"] < t_until]
    te = df[df["state_ts_ms"] >= t_from]
    return tr, te


def fit(train_until: str, test_from: str, sigma_per_sec: float = 5e-4) -> dict:
    df = _load_labeled()
    df = df[df["resolved_up"].notna()].copy()
    df = add_features(df)
    df["p_0"] = fair_prob(df["chainlink_price"], df["open_ref_price"], df["tte_ms"] / 1000.0, sigma_per_sec)
    df = df[df["p_0"].notna()]

    tr, te = _split_by_date(df, train_until, test_from)
    if tr.empty or te.empty:
        raise SystemExit(f"empty split: train={len(tr)}, test={len(te)}")

    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["p_0"], tr["resolved_up"])
    p_star_te = iso.predict(te["p_0"])

    metrics = {
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "brier_raw": float(brier_score_loss(te["resolved_up"], te["p_0"])),
        "brier_iso": float(brier_score_loss(te["resolved_up"], p_star_te)),
        "logloss_raw": float(log_loss(te["resolved_up"], np.clip(te["p_0"], 1e-6, 1 - 1e-6))),
        "logloss_iso": float(log_loss(te["resolved_up"], np.clip(p_star_te, 1e-6, 1 - 1e-6))),
        "auc_raw": _safe_auc(te["resolved_up"], te["p_0"]),
        "auc_iso": _safe_auc(te["resolved_up"], p_star_te),
    }

    # Optional logistic calibrator on a small feature set.
    try:
        logit = LogisticRegression(max_iter=1000).fit(
            tr[FEATURE_COLS].to_numpy(), tr["resolved_up"].to_numpy()
        )
        p_logit_te = logit.predict_proba(te[FEATURE_COLS].to_numpy())[:, 1]
        metrics["brier_logit"] = float(brier_score_loss(te["resolved_up"], p_logit_te))
        metrics["logloss_logit"] = float(log_loss(te["resolved_up"], np.clip(p_logit_te, 1e-6, 1 - 1e-6)))
        metrics["auc_logit"] = _safe_auc(te["resolved_up"], p_logit_te)
    except Exception as e:  # pragma: no cover - defensive
        log.warning("logistic fit failed: %s", e)

    bp = list(zip(iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()))
    out = {
        "sigma_per_sec": sigma_per_sec,
        "isotonic_breakpoints": bp,
        "metrics": metrics,
    }
    path = derived("calibrator.json")
    path.write_text(json.dumps(out, indent=2))
    log.info("wrote calibrator to %s", path)
    log.info("metrics: %s", json.dumps(metrics, indent=2))
    return out


def _safe_auc(y: pd.Series, p: np.ndarray) -> float:
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-until", required=True, help="YYYY-MM-DD")
    ap.add_argument("--test-from", required=True, help="YYYY-MM-DD")
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    args = ap.parse_args()
    fit(args.train_until, args.test_from, args.sigma_per_sec)


if __name__ == "__main__":
    _main()
