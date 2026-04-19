"""Fit the p_0 -> p* calibrator (§11.2).

Supports two splitting strategies:

1. **Leave-One-Market-Out (LOMO)** — default when the dataset has few unique
   markets (< ``--min-markets-for-global``, default 30).  For each market M the
   isotonic regression is trained on all markets *except* M.  This guarantees
   the calibrator never sees the outcome it is predicting.

2. **Global time-ordered split** — used when >= ``min_markets`` are available.
   A single isotonic regression is fit on the first ``fraction`` of data (by
   time) and evaluated on the remainder.

The output is written to ``data/derived/calibrator.json``.  When LOMO is used
the JSON contains a ``lomo_folds`` dict keyed by market_id; the taker backtest
reads this and applies the correct fold per market.

Usage:
    # Fraction-based split (for short test runs):
    python -m raft_research.calibrate --fraction 0.7
    # Date-based split (for multi-day datasets):
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

# If fewer than this many unique markets exist, force LOMO to prevent
# the isotonic regression from memorizing per-market outcomes.
DEFAULT_MIN_MARKETS_FOR_GLOBAL = 30


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


def _split_by_fraction(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered fractional split. Useful for short (sub-day) datasets."""
    df_sorted = df.sort_values("state_ts_ms")
    n = int(len(df_sorted) * fraction)
    return df_sorted.iloc[:n], df_sorted.iloc[n:]


def fit(train_until: str | None = None, test_from: str | None = None,
        sigma_per_sec: float = 5e-4, fraction: float | None = None,
        min_markets_for_global: int = DEFAULT_MIN_MARKETS_FOR_GLOBAL) -> dict:
    df = _load_labeled()
    df = df[df["resolved_up"].notna()].copy()
    df = add_features(df)
    df["p_0"] = fair_prob(df["chainlink_price"], df["open_ref_price"], df["tte_ms"] / 1000.0, sigma_per_sec)
    df = df[df["p_0"].notna()]

    n_markets = df["market_id"].nunique()
    log.info("dataset has %d unique markets (%d rows)", n_markets, len(df))

    # ── Decide: LOMO vs global split ────────────────────────────────────
    if n_markets < min_markets_for_global:
        log.info("n_markets (%d) < %d — using leave-one-market-out calibration",
                 n_markets, min_markets_for_global)
        return _fit_lomo(df, sigma_per_sec)

    # Global time-ordered split.
    if fraction is not None:
        tr, te = _split_by_fraction(df, fraction)
    elif train_until is not None and test_from is not None:
        tr, te = _split_by_date(df, train_until, test_from)
    else:
        raise SystemExit("specify either --fraction or both --train-until and --test-from")
    if tr.empty or te.empty:
        raise SystemExit(f"empty split: train={len(tr)}, test={len(te)}")

    return _fit_global(tr, te, sigma_per_sec, df)


def _fit_lomo(df: pd.DataFrame, sigma_per_sec: float) -> dict:
    """Leave-One-Market-Out: fit a separate isotonic per held-out market."""
    markets = df["market_id"].unique()
    folds: dict = {}
    all_y_true = []
    all_p_raw = []
    all_p_cal = []

    for held_out in markets:
        train_mask = df["market_id"] != held_out
        test_mask = df["market_id"] == held_out
        tr = df[train_mask]
        te = df[test_mask]

        if tr["resolved_up"].nunique() < 2:
            # Cannot fit isotonic without both classes — fall back to raw p_0.
            log.warning("LOMO fold %s: train has only one class, using raw p_0", held_out[:16])
            bp_list: list = []
            p_cal_fold = te["p_0"].to_numpy()
        else:
            iso = IsotonicRegression(out_of_bounds="clip").fit(tr["p_0"], tr["resolved_up"])
            p_cal_fold = iso.predict(te["p_0"])
            bp_list = list(zip(iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()))

        folds[held_out] = {
            "isotonic_breakpoints": bp_list,
            "n_train_rows": int(len(tr)),
            "n_train_markets": int(tr["market_id"].nunique()),
        }
        all_y_true.extend(te["resolved_up"].tolist())
        all_p_raw.extend(te["p_0"].tolist())
        all_p_cal.extend(p_cal_fold.tolist())

    all_y_true = np.asarray(all_y_true)
    all_p_raw = np.asarray(all_p_raw)
    all_p_cal = np.asarray(all_p_cal)

    metrics = {
        "method": "lomo",
        "n_markets": int(len(markets)),
        "n_rows": int(len(df)),
        "brier_raw": float(brier_score_loss(all_y_true, all_p_raw)),
        "brier_iso": float(brier_score_loss(all_y_true, np.clip(all_p_cal, 0, 1))),
        "logloss_raw": float(log_loss(all_y_true, np.clip(all_p_raw, 1e-6, 1 - 1e-6))),
        "logloss_iso": float(log_loss(all_y_true, np.clip(all_p_cal, 1e-6, 1 - 1e-6))),
        "auc_raw": _safe_auc(pd.Series(all_y_true), all_p_raw),
        "auc_iso": _safe_auc(pd.Series(all_y_true), all_p_cal),
    }

    out = {
        "sigma_per_sec": sigma_per_sec,
        "lomo_folds": folds,
        "metrics": metrics,
    }
    path = derived("calibrator.json")
    path.write_text(json.dumps(out, indent=2))
    log.info("wrote LOMO calibrator (%d folds) to %s", len(folds), path)
    log.info("metrics: %s", json.dumps(metrics, indent=2))
    return out


def _fit_global(tr: pd.DataFrame, te: pd.DataFrame, sigma_per_sec: float,
                full_df: pd.DataFrame) -> dict:
    """Global isotonic regression on a time-ordered train/test split."""
    iso = IsotonicRegression(out_of_bounds="clip").fit(tr["p_0"], tr["resolved_up"])
    p_star_te = iso.predict(te["p_0"])

    metrics = {
        "method": "global",
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_train_markets": int(tr["market_id"].nunique()),
        "n_test_markets": int(te["market_id"].nunique()),
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
    ap.add_argument("--train-until", help="YYYY-MM-DD (date-based split)")
    ap.add_argument("--test-from", help="YYYY-MM-DD (date-based split)")
    ap.add_argument("--fraction", type=float, help="Fractional train split 0-1 (time-ordered)")
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--min-markets-for-global", type=int, default=DEFAULT_MIN_MARKETS_FOR_GLOBAL,
                    help="Use LOMO when fewer than this many markets exist (default: 30)")
    args = ap.parse_args()
    fit(args.train_until, args.test_from, args.sigma_per_sec, args.fraction,
        args.min_markets_for_global)


if __name__ == "__main__":
    _main()
