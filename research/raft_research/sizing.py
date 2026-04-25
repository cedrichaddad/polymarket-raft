"""Position-sizing for the frozen taker rule_gate.

Three modes, in order of increasing aggressiveness:

1. ``flat`` — every accepted trade gets size = 1.0. Baseline for comparison.
2. ``bucketed`` — discrete size multipliers keyed off post-cost edge, tte
   bucket, and quote-quality penalties. Safe and interpretable.
3. ``fractional_kelly`` — shrunk probability → Kelly → clipped to a small
   fraction. Designed to be small, not optimal.

Portfolio caps are enforced chronologically after per-trade sizes are set:

* ``per_market_cap`` — cap on one trade's bankroll fraction.
* ``gross_exposure_cap`` — cap on sum of open |bankroll_frac| at any moment.
* ``net_exposure_cap`` — cap on signed sum (buy positive, sell negative).
* ``max_concurrent_trades`` — cap on number of trades open simultaneously.

"Open" means: trade entered at ``state_ts_ms`` and has not yet reached its
market's ``end_ts_ms``. Sizes that would violate a cap are scaled down
proportionally; if scaling to zero would be required, the trade is dropped
from the sized run (``size=0.0``).

The module returns the trade frame with three new columns:

* ``raw_size`` — size before caps (from the chosen mode).
* ``size`` — size after caps.
* ``sized_net_pnl`` — ``size * net_pnl``.

Importantly, ``size`` has no monetary unit; it is a bankroll fraction for
Kelly and a notional multiplier for flat/bucketed. ``sized_net_pnl`` is in
the same units as ``net_pnl`` (per one "YES" share), scaled by size.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Iterable

import numpy as np
import pandas as pd


# ─── config ────────────────────────────────────────────────────────────────

SIZING_MODES: tuple[str, ...] = (
    "flat",
    "bucketed",                        # alias for bucketed_full (back-compat)
    "bucketed_full",
    "bucketed_no_persistence_penalty",
    "bucketed_no_spread_penalty",
    "bucketed_no_bucket_ceiling",
    "bucketed_edge_only",
    "fractional_kelly",
    "frac_kelly",                      # alias for fractional_kelly
)


@dataclass
class SizingConfig:
    """Knobs for all sizing modes.

    The default values are deliberately conservative: Kelly is 20 % with 0.4
    shrinkage, per-market cap at 1 % of bankroll, gross cap at 3 %.

    Bucketed ablation variants map onto the three boolean switches below.
    `mode` is the authoritative selector; when mode starts with `bucketed_*`
    the switches are configured to match the ablation.
    """
    mode: str = "flat"

    # Costs to subtract from edge when computing post-cost edge.
    fee: float = 0.0
    slippage_haircut: float = 0.0

    # --- bucketed mode + ablation switches ---
    apply_persistence_penalty: bool = True
    apply_spread_penalty: bool = True
    apply_bucket_ceiling: bool = True

    #   (edge_net_lo, edge_net_hi, size_mult)
    bucketed_levels: list[tuple[float, float, float]] = field(default_factory=lambda: [
        (0.01, 0.02, 0.25),
        (0.02, 0.04, 0.50),
        (0.04, 0.06, 0.75),
        (0.06, 1.00, 1.00),
    ])
    # Additive penalties (subtract from final multiplier).
    wide_spread_threshold: float = 0.03
    wide_spread_penalty: float = 0.25
    weak_persistence_penalty: float = 0.25
    # tte-bucket size ceilings: per bucket, a max multiplier. None = no cap.
    bucket_ceiling: dict[str, float] = field(default_factory=lambda: {
        "15_30": 0.0,   # disabled — matches the frozen rule_gate
        "10_15": 1.0,
        "5_10":  0.75,
        "2_5":   0.50,
        "0_2":   0.0,   # disabled — execution risk
    })

    # --- fractional Kelly ---
    kelly_shrinkage: float = 0.4       # λ: p_shrunk = 0.5 + λ*(p - 0.5)
    kelly_fraction: float = 0.2        # α: size = α * f_kelly
    kelly_cap_per_trade: float = 0.01  # redundant with per_market_cap; belt-and-braces

    # --- portfolio caps ---
    per_market_cap: float = 0.01          # 1 % bankroll per market
    gross_exposure_cap: float = 0.03      # 3 % gross
    net_exposure_cap: float = 0.02        # 2 % net directional
    max_concurrent_trades: int = 4

    def to_dict(self) -> dict:
        return asdict(self)


# ─── raw size (per-trade, pre-cap) ─────────────────────────────────────────

def _edge_net(df: pd.DataFrame, cfg: SizingConfig) -> pd.Series:
    """|edge| after subtracting fee + slippage_haircut. Clipped at 0."""
    abs_edge = df["abs_edge"] if "abs_edge" in df.columns else (df["p_hybrid"] - df["market_prob"]).abs()
    return (abs_edge - cfg.fee - cfg.slippage_haircut).clip(lower=0.0)


def _raw_flat(df: pd.DataFrame, cfg: SizingConfig) -> np.ndarray:
    return np.ones(len(df), dtype=float)


def _raw_bucketed(df: pd.DataFrame, cfg: SizingConfig) -> np.ndarray:
    edge_net = _edge_net(df, cfg).to_numpy()
    mult = np.zeros(len(df), dtype=float)
    for lo, hi, m in cfg.bucketed_levels:
        mask = (edge_net >= lo) & (edge_net < hi)
        mult[mask] = m
    # Spread penalty (ablation switch).
    if cfg.apply_spread_penalty and "spread_yes" in df.columns:
        mult = np.where(
            df["spread_yes"].fillna(0).to_numpy() > cfg.wide_spread_threshold,
            mult - cfg.wide_spread_penalty, mult,
        )
    # Persistence penalty (ablation switch).
    if cfg.apply_persistence_penalty and "edge_persist_2s" in df.columns:
        mult = np.where(df["edge_persist_2s"].fillna(False).to_numpy().astype(bool),
                        mult, mult - cfg.weak_persistence_penalty)
    mult = np.clip(mult, 0.0, 1.0)
    # tte-bucket ceiling (ablation switch).
    if cfg.apply_bucket_ceiling and "tte_bucket" in df.columns:
        ceil = df["tte_bucket"].map(cfg.bucket_ceiling).fillna(0.0).to_numpy()
        mult = np.minimum(mult, ceil)
    return mult


def _raw_kelly(df: pd.DataFrame, cfg: SizingConfig) -> np.ndarray:
    p = df["p_hybrid"].to_numpy()
    c = df["fill_price"].to_numpy() if "fill_price" in df.columns else df["market_prob"].to_numpy()
    side_is_buy = df["side_is_buy"].to_numpy().astype(bool)

    # Effective win prob from our perspective: buy wins when resolved up.
    p_eff = np.where(side_is_buy, p, 1.0 - p)
    c_eff = np.where(side_is_buy, c, 1.0 - c)

    # Shrink toward 0.5.
    p_shrunk = 0.5 + cfg.kelly_shrinkage * (p_eff - 0.5)

    # Binary-contract Kelly: f = (p - c) / (1 - c)  (valid for c in (0,1)).
    denom = np.clip(1.0 - c_eff, 1e-4, 1.0)
    f_kelly = np.clip((p_shrunk - c_eff) / denom, 0.0, 1.0)

    sized = cfg.kelly_fraction * f_kelly
    return np.minimum(sized, cfg.kelly_cap_per_trade)


_RAW_FN = {
    "flat": _raw_flat,
    "bucketed": _raw_bucketed,
    "bucketed_full": _raw_bucketed,
    "bucketed_no_persistence_penalty": _raw_bucketed,
    "bucketed_no_spread_penalty": _raw_bucketed,
    "bucketed_no_bucket_ceiling": _raw_bucketed,
    "bucketed_edge_only": _raw_bucketed,
    "fractional_kelly": _raw_kelly,
    "frac_kelly": _raw_kelly,
}


# Ablation-switch overrides applied in `make_config()`. Only the three
# switches change; every bucketed variant shares the same edge levels,
# thresholds, and caps. The dev mean changes only because of the switch.
_ABLATION_SWITCHES: dict[str, dict[str, bool]] = {
    "flat": {},
    "bucketed": {},
    "bucketed_full": {},
    "bucketed_no_persistence_penalty": {"apply_persistence_penalty": False},
    "bucketed_no_spread_penalty": {"apply_spread_penalty": False},
    "bucketed_no_bucket_ceiling": {"apply_bucket_ceiling": False},
    "bucketed_edge_only": {
        "apply_persistence_penalty": False,
        "apply_spread_penalty": False,
        "apply_bucket_ceiling": False,
    },
    "fractional_kelly": {},
    "frac_kelly": {},
}


def make_config(mode: str, fee: float = 0.0, slippage_haircut: float = 0.0) -> "SizingConfig":
    """Factory that maps a mode name to a SizingConfig with correct switches."""
    if mode not in SIZING_MODES:
        raise SystemExit(f"unknown sizing mode: {mode}. Known: {sorted(SIZING_MODES)}")
    cfg = SizingConfig(mode=mode, fee=fee, slippage_haircut=slippage_haircut)
    for attr, val in _ABLATION_SWITCHES[mode].items():
        setattr(cfg, attr, val)
    return cfg


# ─── portfolio-cap sweep ───────────────────────────────────────────────────

def _apply_caps(df: pd.DataFrame, cfg: SizingConfig) -> np.ndarray:
    """Walk chronologically by entry time; scale down sizes that break caps.

    Requires ``state_ts_ms`` (entry) and ``end_ts_ms`` (market close) in df.
    """
    if df.empty:
        return np.zeros(0, dtype=float)
    order = df.sort_values("state_ts_ms").index.to_list()
    raw = df["raw_size"].to_numpy().copy()
    sizes = np.minimum(raw, cfg.per_market_cap) if cfg.mode in ("fractional_kelly", "frac_kelly") else raw.copy()
    # For non-Kelly modes, raw_size is a notional multiplier, not a bankroll
    # fraction, so per_market_cap does not apply. We still enforce the
    # portfolio caps on an equivalent "bankroll fraction" basis where each
    # notional unit is treated as 1 *unit* of exposure and the caps below
    # are expressed in the same unit.
    if cfg.mode != "fractional_kelly":
        sizes = np.minimum(sizes, 1.0)

    start = df["state_ts_ms"].to_numpy()
    end = df["end_ts_ms"].to_numpy() if "end_ts_ms" in df.columns else start + 60_000
    side = df["side_is_buy"].to_numpy().astype(bool)

    # Book of open positions: list of (close_ts_ms, size, side_is_buy).
    final = np.zeros(len(df), dtype=float)
    idx_pos = {idx: i for i, idx in enumerate(df.index)}
    open_book: list[tuple[int, float, bool]] = []

    for idx in order:
        i = idx_pos[idx]
        now = start[i]
        # Evict closed positions.
        open_book = [b for b in open_book if b[0] > now]
        # Remaining headroom.
        gross_used = sum(abs(b[1]) for b in open_book)
        net_used = sum(b[1] if b[2] else -b[1] for b in open_book)
        gross_left = max(0.0, cfg.gross_exposure_cap - gross_used)
        # Net cap applies on the same side as the new trade.
        side_sign = 1.0 if side[i] else -1.0
        net_projection_headroom = cfg.net_exposure_cap - side_sign * net_used
        net_left = max(0.0, net_projection_headroom)

        if len(open_book) >= cfg.max_concurrent_trades:
            final[i] = 0.0
            continue

        # For flat/bucketed modes, the "caps" above are in units of notional
        # multiplier; interpret gross_exposure_cap=3% as a *relative* cap
        # meaning "max 3 notional units open at once" — i.e. scale caps by
        # 100x so flat=1.0 trades aren't pre-capped to near-zero. This keeps
        # flat-mode behavior intuitive while still enforcing concurrency.
        gross_cap_eff = gross_left if cfg.mode in ("fractional_kelly", "frac_kelly") else gross_left * 100.0
        net_cap_eff = net_left if cfg.mode in ("fractional_kelly", "frac_kelly") else net_left * 100.0

        allowed = min(sizes[i], gross_cap_eff, net_cap_eff)
        allowed = max(0.0, allowed)
        final[i] = allowed
        if allowed > 0:
            open_book.append((int(end[i]), allowed, bool(side[i])))
    return final


# ─── public API ────────────────────────────────────────────────────────────

def compute_sizes(trades: pd.DataFrame, cfg: SizingConfig) -> pd.DataFrame:
    """Return a copy of trades with ``raw_size``, ``size``, ``sized_net_pnl``."""
    if trades.empty:
        return trades.assign(raw_size=[], size=[], sized_net_pnl=[])
    fn = _RAW_FN.get(cfg.mode)
    if fn is None:
        raise SystemExit(f"unknown sizing mode: {cfg.mode}")
    out = trades.copy()
    out["raw_size"] = fn(out, cfg)
    out["size"] = _apply_caps(out, cfg)
    out["sized_net_pnl"] = out["size"] * out["net_pnl"]
    return out


# ─── metrics ───────────────────────────────────────────────────────────────

def metrics_bundle(trades: pd.DataFrame, pnl_col: str = "sized_net_pnl") -> dict:
    """PnL + t-stat + drawdown + worst-streak + ret/dd."""
    if trades.empty or trades[pnl_col].abs().sum() == 0:
        return {"n_trades": int(len(trades)), "n_active": 0,
                "total": 0.0, "mean": 0.0, "std": 0.0, "t_stat": None,
                "hit_rate": 0.0, "max_drawdown": 0.0,
                "worst_losing_streak": 0, "ret_over_dd": None}
    active = trades[trades["size"] > 0] if "size" in trades.columns else trades
    pnl = active[pnl_col].to_numpy()
    order = active.sort_values("state_ts_ms")[pnl_col].to_numpy()
    cum = np.cumsum(order)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(drawdowns.max()) if len(drawdowns) else 0.0

    # Worst losing streak (consecutive negatives).
    streak = worst = 0
    for x in order:
        if x < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0

    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0
    t_stat = mean / (std / np.sqrt(len(pnl))) if std > 0 else None
    total = float(pnl.sum())
    hit_rate = float((pnl > 0).mean())
    ret_over_dd = total / max_dd if max_dd > 0 else None

    return {
        "n_trades": int(len(active)),
        "n_active": int((active.get("size", pd.Series([1]*len(active))) > 0).sum()),
        "total": total,
        "mean": mean,
        "std": std,
        "t_stat": float(t_stat) if t_stat is not None else None,
        "hit_rate": hit_rate,
        "max_drawdown": max_dd,
        "worst_losing_streak": int(worst),
        "ret_over_dd": float(ret_over_dd) if ret_over_dd is not None else None,
    }


__all__ = ["SizingConfig", "compute_sizes", "metrics_bundle"]
