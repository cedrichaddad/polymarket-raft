"""Compare model variants (§D.1).

Reads the labeled state table, evaluates:
    - parametric only (p_0)
    - isotonic-calibrated (p*)
    - hybrid with retrieval overlay (if index present)

Writes a compact CSV of Brier / LogLoss / AUC for each variant.
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .fair_value import fair_prob
from .features import FEATURE_COLS, add_features
from .paths import derived

log = logging.getLogger(__name__)


def compare(sigma_per_sec: float = 5e-4) -> pd.DataFrame:
    df = pd.read_parquet(derived("market_state_1s_labeled.parquet"))
    df = df[df["resolved_up"].notna()].copy()
    df = add_features(df)
    df["p_0"] = fair_prob(df["chainlink_price"], df["open_ref_price"], df["tte_ms"] / 1000.0, sigma_per_sec)
    df = df[df["p_0"].notna()]

    cal_path = derived("calibrator.json")
    if cal_path.exists():
        data = json.loads(cal_path.read_text())
        if "lomo_folds" in data:
            # LOMO calibrator: apply per-market-fold breakpoints.
            p_star = np.full(len(df), np.nan, dtype=float)
            for mid, fold in data["lomo_folds"].items():
                mask = df["market_id"].to_numpy() == mid
                if not mask.any():
                    continue
                bp = np.asarray(fold["isotonic_breakpoints"])
                if len(bp) == 0:
                    p_star[mask] = df["p_0"].to_numpy()[mask]
                else:
                    p_star[mask] = np.interp(df["p_0"].to_numpy()[mask], bp[:, 0], bp[:, 1])
            remaining = np.isnan(p_star)
            p_star[remaining] = df["p_0"].to_numpy()[remaining]
            df["p_star"] = p_star
        else:
            bp = np.asarray(data["isotonic_breakpoints"])
            df["p_star"] = np.interp(df["p_0"].to_numpy(), bp[:, 0], bp[:, 1])
    else:
        df["p_star"] = df["p_0"]

    variants = [("p_0", df["p_0"]), ("p_star", df["p_star"])]

    # Optional hybrid with retrieval (if hnswlib index + meta exist).
    idx_path = derived("retrieval.bin")
    meta_path = derived("retrieval_meta.parquet")
    if idx_path.exists() and meta_path.exists():
        try:
            import hnswlib
            meta = pd.read_parquet(meta_path)
            index = hnswlib.Index(space="l2", dim=len(FEATURE_COLS))
            index.load_index(str(idx_path), max_elements=len(meta))
            index.set_ef(128)
            X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
            ids, _ = index.knn_query(X, k=32)
            p_nn = np.nanmean(meta["resolved_up"].to_numpy()[ids], axis=1)
            tte_norm = df["tte_ms"].to_numpy() / (df["window_type"].map({"btc_5m": 300_000, "btc_15m": 900_000}).fillna(300_000).to_numpy())
            w_param = 0.5 + 0.5 * np.clip(tte_norm, 0, 1)
            df["p_hybrid"] = w_param * df["p_star"].to_numpy() + (1 - w_param) * p_nn
            variants.append(("p_hybrid", df["p_hybrid"]))
        except Exception as e:  # pragma: no cover - defensive
            log.warning("hybrid eval failed: %s", e)

    rows = []
    for name, p in variants:
        p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
        rows.append({
            "variant": name,
            "brier": float(brier_score_loss(df["resolved_up"], p_clipped)),
            "logloss": float(log_loss(df["resolved_up"], p_clipped)),
            "auc": float(roc_auc_score(df["resolved_up"], p_clipped)) if df["resolved_up"].nunique() == 2 else np.nan,
        })
    result = pd.DataFrame(rows)
    out = derived("compare_models.csv")
    result.to_csv(out, index=False)
    log.info("wrote %s", out)
    log.info("\n%s", result.to_string(index=False))
    return result


def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    args = ap.parse_args()
    compare(args.sigma_per_sec)


if __name__ == "__main__":
    _main()
