"""Feature vector construction in Python (§10.1, Appendix C).

Mirrors the Rust `state_builder` logic so retrieval indexes built in Python
stay compatible with the live service's query vectors.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "z_open",
    "z_short",
    "tte_norm",
    "spread_yes",
    "imbalance_yes",
    "signed_flow_1s",
    "signed_flow_5s",
    "rv_10s",
    "rv_30s",
    "chainlink_binance_gap",
    "tick_regime",
    "market_prob",
]


def add_features(
    state: pd.DataFrame,
    horizon_seconds: float | None = None,
) -> pd.DataFrame:
    """Attach the §10.1 feature columns.

    `state` must already contain the output of `build_labels` (i.e. have
    `chainlink_price`, `binance_price`, `mid_yes`, `spread_yes`,
    `signed_flow_1s`, `open_ref_price`, `tte_ms`, `window_type`).
    """
    df = state.sort_values(["market_id", "state_ts_ms"]).copy()

    horizon_col = horizon_seconds or df["window_type"].map(
        {"btc_5m": 300.0, "btc_15m": 900.0}
    ).fillna(300.0)
    df["tte_norm"] = (df["tte_ms"] / 1000.0 / horizon_col).clip(0.0, 1.0)

    # Rolling Chainlink-based realized vol and short-horizon returns.
    df["chain_log"] = np.log(df["chainlink_price"].replace(0, np.nan))
    df["chain_ret_1s"] = df.groupby("market_id")["chain_log"].diff()
    df["rv_10s"] = df.groupby("market_id")["chain_ret_1s"].rolling(10, min_periods=3).std().reset_index(level=0, drop=True)
    df["rv_30s"] = df.groupby("market_id")["chain_ret_1s"].rolling(30, min_periods=5).std().reset_index(level=0, drop=True)
    df["rv_10s"] = df["rv_10s"].fillna(0.0)
    df["rv_30s"] = df["rv_30s"].fillna(0.0)

    df["z_short"] = df.groupby("market_id")["chain_log"].diff(5) / df["rv_10s"].replace(0, np.nan)
    df["z_short"] = df["z_short"].fillna(0.0)

    # z_open: log(S/K) / (rv_30s * sqrt(tte+1)).
    tte_s = (df["tte_ms"] / 1000.0).clip(lower=0.0)
    denom = (df["rv_30s"] * np.sqrt(tte_s + 1.0)).replace(0, np.nan)
    df["z_open"] = (np.log(df["chainlink_price"] / df["open_ref_price"]) / denom).fillna(0.0)

    # Signed flow 5s rolling sum.
    df["signed_flow_5s"] = (
        df.groupby("market_id")["signed_flow_1s"].rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    df["signed_flow_5s"] = df["signed_flow_5s"].fillna(0.0)

    # L1 imbalance proxy.
    tot = df["best_yes_bid"] + df["best_yes_ask"]
    df["imbalance_yes"] = np.where(
        tot > 0, (df["best_yes_bid"] - df["best_yes_ask"]) / tot, 0.0
    )

    # Market prob + tick regime (center vs wing).
    df["market_prob"] = df["mid_yes"].clip(0.0, 1.0)
    df["tick_regime"] = (np.abs(df["market_prob"] - 0.5) <= 0.1).astype(float)

    df["chainlink_binance_gap"] = df["chainlink_price"] - df["binance_price"]
    df["chainlink_binance_gap"] = df["chainlink_binance_gap"].fillna(0.0)

    # Drop scratch columns; keep only inputs + features + label.
    df = df.drop(columns=["chain_log", "chain_ret_1s"])
    return df
