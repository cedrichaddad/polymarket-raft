"""Time-to-expiry-aware taker entry gate.

Two pieces:

1. `add_gate_features(df)` — given a dataframe that already has the output of
   `add_features()` plus `p_hybrid` and `market_prob`, attach the gate-specific
   features: edge, bucket, persistence, slope, mid-change, quote-quality flag.

2. `select_taker_entries(df, gate_config, first_trigger_only=True)` — apply a
   rule-based gate. Candidate rows must satisfy
       * `quote_quality_flag` is True,
       * `abs_edge >= threshold_by_bucket[tte_bucket]`,
       * persistence requirement (edge on same side and above threshold for N bars).
   Returns a trade-level dataframe (one row per accepted entry), with
   first-trigger-per-market deduplication when `first_trigger_only=True`.

Design notes:
    - The gate is a layer on top of the existing p_hybrid. It does not replace the
      underlying fair-value model.
    - Buckets are labeled by their (min, max) tte in seconds. The "0_2" bucket is
      intentionally available but disabled by default: the last ~2s is the most
      execution-sensitive and the least informative after a conservative fill.
    - All thresholds and which buckets are enabled are configurable so we can
      sweep without code changes.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd


# Bucket edges in seconds, closed on the low side, open on the high side except
# the last bucket which is closed-closed. (min, max, label)
DEFAULT_BUCKETS: list[tuple[float, float, str]] = [
    (15.0, 30.0, "15_30"),
    (10.0, 15.0, "10_15"),
    (5.0,  10.0, "5_10"),
    (2.0,  5.0,  "2_5"),
    (0.0,  2.0,  "0_2"),
]


@dataclass
class GateConfig:
    """Configuration for the rule-based taker gate."""
    max_entry_window_secs: float = 15.0
    persist_bars: int = 2
    require_persistence: bool = True
    # Per-bucket absolute edge threshold. None disables the bucket.
    thresholds: dict[str, float | None] = field(default_factory=lambda: {
        "15_30": None,    # disabled outside the entry window by default
        "10_15": 0.03,
        "5_10":  0.025,
        "2_5":   0.02,
        "0_2":   None,    # disabled by default (execution risk)
    })
    # Quote-quality gate parameters.
    max_spread: float = 0.05
    max_chainlink_binance_gap_abs: float = 500.0  # USD

    def thresholds_with_defaults(self) -> dict[str, float | None]:
        out = {b[2]: None for b in DEFAULT_BUCKETS}
        out.update(self.thresholds)
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "GateConfig":
        return cls(
            max_entry_window_secs=d.get("max_entry_window_secs", 15.0),
            persist_bars=d.get("persist_bars", 2),
            require_persistence=d.get("require_persistence", True),
            thresholds=d.get("thresholds", {}),
            max_spread=d.get("max_spread", 0.05),
            max_chainlink_binance_gap_abs=d.get("max_chainlink_binance_gap_abs", 500.0),
        )

    def to_dict(self) -> dict:
        return {
            "max_entry_window_secs": self.max_entry_window_secs,
            "persist_bars": self.persist_bars,
            "require_persistence": self.require_persistence,
            "thresholds": self.thresholds,
            "max_spread": self.max_spread,
            "max_chainlink_binance_gap_abs": self.max_chainlink_binance_gap_abs,
        }


def _bucket_tte(tte_s: pd.Series, buckets: Iterable[tuple[float, float, str]] = DEFAULT_BUCKETS) -> pd.Series:
    """Map tte seconds to bucket label. Returns `""` for rows outside all buckets.

    Convention: `[low, high)` for all buckets, with the top bucket capped at
    `<= high`. Rows at exactly the boundary go to the **lower-numbered** bucket
    (e.g. tte_s=15 lands in `10_15`, not `15_30`), so the default 15s entry
    window classifies the boundary row as "late window".
    """
    out = pd.Series([""] * len(tte_s), index=tte_s.index, dtype=object)
    vals = tte_s.to_numpy()
    for lo, hi, label in buckets:
        if label == "15_30":
            mask = (vals > lo) & (vals <= hi)       # (15, 30]
        elif label == "10_15":
            mask = (vals >= lo) & (vals <= hi)      # [10, 15]
        else:
            mask = (vals >= lo) & (vals < hi)
        out.loc[mask] = label
    return out


def add_gate_features(df: pd.DataFrame, gate_config: GateConfig | None = None) -> pd.DataFrame:
    """Attach gate-specific feature columns.

    Requires: `p_hybrid`, `market_prob`, `tte_ms`, `spread_yes`, and the columns
    produced by `features.add_features` (`chainlink_binance_gap`, `rv_*`, etc.).
    """
    cfg = gate_config or GateConfig()
    df = df.sort_values(["market_id", "state_ts_ms"]).copy()

    df["tte_s"] = df["tte_ms"] / 1000.0
    df["tte_bucket"] = _bucket_tte(df["tte_s"])

    df["edge_raw"] = df["p_hybrid"] - df["market_prob"]
    df["abs_edge"] = df["edge_raw"].abs()
    df["signal_side"] = np.where(df["edge_raw"] > 0, "buy", "sell")

    # Per-market rolling diagnostics. A bar is "on same side and above threshold"
    # if `abs_edge > 0` AND `edge_raw` sign equals the current sign. We capture
    # persistence at two horizons so the gate can be tuned.
    g = df.groupby("market_id", sort=False)
    sign = np.sign(df["edge_raw"].fillna(0.0)).astype(int)
    df["edge_sign"] = sign

    def _persist(n_bars: int) -> pd.Series:
        # Returns True if the last n bars (inclusive of current) all have
        # matching nonzero sign.
        by_mkt = sign.groupby(df["market_id"], sort=False)
        # cumulative: same-sign streak length — reset whenever sign changes.
        def streak(s: pd.Series) -> pd.Series:
            change = (s != s.shift()).astype(int)
            grp = change.cumsum()
            return s.groupby(grp).cumcount() + 1
        lens = by_mkt.transform(streak)
        return (lens >= n_bars) & (sign != 0)

    df["edge_persist_2s"] = _persist(2)
    df["edge_persist_3s"] = _persist(3)

    df["edge_slope_2s"] = g["abs_edge"].diff(2).fillna(0.0)

    # Mid movement proxies.
    df["mid_change_1s"] = g["mid_yes"].diff(1).fillna(0.0)
    df["mid_change_2s"] = g["mid_yes"].diff(2).fillna(0.0)

    # Quote-quality flag.
    q_spread = df["spread_yes"].fillna(1.0) <= cfg.max_spread
    q_gap = df["chainlink_binance_gap"].abs().fillna(1e9) <= cfg.max_chainlink_binance_gap_abs
    q_have_prices = df["chainlink_price"].notna() & df["mid_yes"].notna()
    df["quote_quality_flag"] = q_spread & q_gap & q_have_prices

    return df


def select_taker_entries(
    df: pd.DataFrame,
    gate_config: GateConfig,
    first_trigger_only: bool = True,
) -> pd.DataFrame:
    """Apply the rule-based gate.

    Expects `df` to already include gate features (call `add_gate_features` first).
    Returns a frame of accepted entry rows. If `first_trigger_only` is True, keeps
    only the earliest accepting row per market.
    """
    thresholds = gate_config.thresholds_with_defaults()

    # Entry window filter.
    cand = df[(df["tte_s"] > 0) & (df["tte_s"] <= gate_config.max_entry_window_secs)].copy()

    # Quote quality.
    cand = cand[cand["quote_quality_flag"]].copy()

    # Per-bucket threshold.
    def _passes_threshold(row) -> bool:
        t = thresholds.get(row["tte_bucket"])
        if t is None:
            return False
        return bool(row["abs_edge"] >= t)

    if not cand.empty:
        mask = cand.apply(_passes_threshold, axis=1)
        cand = cand[mask].copy()

    # Persistence.
    if gate_config.require_persistence and not cand.empty:
        col = "edge_persist_2s" if gate_config.persist_bars <= 2 else "edge_persist_3s"
        cand = cand[cand[col]].copy()

    if first_trigger_only and not cand.empty:
        cand = cand.sort_values(["market_id", "state_ts_ms"])
        cand = cand.groupby("market_id", sort=False).first().reset_index()

    return cand


__all__ = [
    "DEFAULT_BUCKETS",
    "GateConfig",
    "add_gate_features",
    "select_taker_entries",
]
