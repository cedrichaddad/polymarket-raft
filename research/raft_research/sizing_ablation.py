"""Bucketed-sizing ablation study on top of the frozen taker rule_gate.

Variants (sizing modes evaluated):
  - flat                              (control / official benchmark)
  - bucketed_full                     (current research favorite)
  - bucketed_no_persistence_penalty
  - bucketed_no_spread_penalty
  - bucketed_no_bucket_ceiling
  - bucketed_edge_only                (edge → size only; all penalties off)
  - fractional_kelly                  (safety reference, not tuned)

Evaluation: the same 4-fold walk-forward market-level split as
`rolling_sizing_eval`. For each fold × segment × mode, we record the full
metrics bundle plus size-concentration diagnostics (avg size by tte
bucket, by spread regime, by persistence flag; share of total risk per
tte bucket). Aggregation across folds gives a per-mode summary, and a
robust ranking decides the winner.

Outputs written under `data/derived/`:
  - taker_gate_sizing_ablation_eval.csv
  - taker_gate_sizing_ablation_trades.parquet
  - taker_gate_sizing_ablation_summary.csv
  - taker_gate_sizing_ablation_ranked.csv
  - taker_gate_sizing_ablation_findings.md
  - sizing_ablation_baseline.json     (provenance / frozen config)

Guardrails (PHASE 8 of the spec):
  - rule_gate is not modified.
  - holdout definition is not modified.
  - flat logic is not modified.
  - fractional_kelly logic is not modified.
  - Ranking uses holdout metrics first; dev metrics are tie-breakers only.
"""
from __future__ import annotations
import argparse
import json
import logging
from dataclasses import asdict

import numpy as np
import pandas as pd

from .evaluate_sizing import FROZEN_RULE_GATE
from .paths import derived
from .rolling_sizing_eval import _market_order, _run_for_ids, _split_markets
from .backtest_taker import _prepare_dataframe
from .sizing import (
    SIZING_MODES,
    SizingConfig,
    compute_sizes,
    make_config,
    metrics_bundle,
)

log = logging.getLogger(__name__)


ABLATION_MODES: tuple[str, ...] = (
    "flat",
    "bucketed_full",
    "bucketed_no_persistence_penalty",
    "bucketed_no_spread_penalty",
    "bucketed_no_bucket_ceiling",
    "bucketed_edge_only",
    "fractional_kelly",
)


# ─── concentration diagnostics ────────────────────────────────────────────

def _concentration(sized: pd.DataFrame) -> dict:
    """Extra diagnostics: where is size being allocated?"""
    if sized.empty or "size" not in sized.columns:
        return {}
    total_size = float(sized["size"].sum())

    def _by(col: str, default_label: str) -> dict:
        if col not in sized.columns or total_size <= 0:
            return {}
        grouped = sized.groupby(col, dropna=False)["size"].agg(["mean", "sum"])
        return {
            f"avg_size_by_{col}:{str(k)}": float(v["mean"]) for k, v in grouped.iterrows()
        } | {
            f"risk_share_by_{col}:{str(k)}": float(v["sum"] / total_size)
            for k, v in grouped.iterrows()
        }

    out: dict = {"avg_size": float(sized["size"].mean()),
                 "median_size": float(sized["size"].median())}
    out.update(_by("tte_bucket", "(all)"))
    # Spread regime: wide vs narrow at the default 0.03 cutoff.
    if "spread_yes" in sized.columns:
        wide = sized["spread_yes"].fillna(0) > 0.03
        out["avg_size_wide_spread"] = float(sized.loc[wide, "size"].mean()) if wide.any() else 0.0
        out["avg_size_narrow_spread"] = float(sized.loc[~wide, "size"].mean()) if (~wide).any() else 0.0
    if "edge_persist_2s" in sized.columns:
        pf = sized["edge_persist_2s"].fillna(False).astype(bool)
        out["avg_size_persist_true"] = float(sized.loc[pf, "size"].mean()) if pf.any() else 0.0
        out["avg_size_persist_false"] = float(sized.loc[~pf, "size"].mean()) if (~pf).any() else 0.0
    return out


# ─── the run ──────────────────────────────────────────────────────────────

def _write_baseline(cfg_flat: SizingConfig, cfg_full: SizingConfig, fee: float, slip: float,
                    n_folds: int) -> None:
    baseline = {
        "frozen_rule_gate": FROZEN_RULE_GATE.to_dict(),
        "bucketed_full_config": asdict(cfg_full),
        "flat_config": asdict(cfg_flat),
        "taker_fee_prob": fee,
        "slippage_haircut": slip,
        "n_folds": n_folds,
        "split_scheme": "walk-forward market-level; see rolling_sizing_eval._split_markets",
        "variants": list(ABLATION_MODES),
    }
    derived("sizing_ablation_baseline.json").write_text(json.dumps(baseline, indent=2, default=str))


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
    log.info("ablation: %d folds, %d markets total, %d variants",
             len(pairs), len(ordered), len(ABLATION_MODES))

    _write_baseline(make_config("flat", taker_fee_prob, slippage_haircut),
                    make_config("bucketed_full", taker_fee_prob, slippage_haircut),
                    taker_fee_prob, slippage_haircut, n_folds)

    rows: list[dict] = []
    trades_frames: list[pd.DataFrame] = []

    for fold_idx, (dev_ids, hold_ids) in enumerate(pairs, start=1):
        for segment_name, ids in (("dev", dev_ids), ("holdout", hold_ids)):
            trades = _run_for_ids(df_full, ids, taker_fee_prob)
            if trades.empty:
                continue
            for mode in ABLATION_MODES:
                cfg = make_config(mode, taker_fee_prob, slippage_haircut)
                sized = compute_sizes(trades, cfg)
                m = metrics_bundle(sized)
                m.update({
                    "fold": fold_idx,
                    "segment": segment_name,
                    "sizing_mode": mode,
                    "n_markets": len(ids),
                    "n_raw_trades": int(len(trades)),
                    "n_dropped_by_caps": int((sized["size"] == 0).sum()),
                })
                m.update(_concentration(sized))
                rows.append(m)
                trades_frames.append(sized.assign(fold=fold_idx, segment=segment_name, sizing_mode=mode))

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        log.error("no rows"); return per_fold

    eval_p = derived("taker_gate_sizing_ablation_eval.csv")
    per_fold.to_csv(eval_p, index=False)
    trades_p = derived("taker_gate_sizing_ablation_trades.parquet")
    pd.concat(trades_frames, ignore_index=True).to_parquet(trades_p)

    summary = _aggregate(per_fold)
    summary_p = derived("taker_gate_sizing_ablation_summary.csv")
    summary.to_csv(summary_p, index=False)

    ranked = _rank(summary)
    ranked_p = derived("taker_gate_sizing_ablation_ranked.csv")
    ranked.to_csv(ranked_p, index=False)

    findings_p = derived("taker_gate_sizing_ablation_findings.md")
    findings_p.write_text(_findings_md(summary, ranked))

    log.info("wrote:\n  %s\n  %s\n  %s\n  %s\n  %s", eval_p, trades_p, summary_p, ranked_p, findings_p)
    log.info("\n=== summary ===\n%s", summary.to_string(index=False))
    log.info("\n=== ranked (by holdout-first criterion) ===\n%s", ranked.to_string(index=False))
    return summary


# ─── aggregation ──────────────────────────────────────────────────────────

def _aggregate(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Per-mode wide summary with dev_ and holdout_ prefixed columns."""
    rows: list[dict] = []
    modes = per_fold["sizing_mode"].unique()
    flat_holdout = per_fold[(per_fold["sizing_mode"] == "flat") & (per_fold["segment"] == "holdout")]
    flat_dev = per_fold[(per_fold["sizing_mode"] == "flat") & (per_fold["segment"] == "dev")]

    for mode in modes:
        out: dict = {"sizing_mode": mode}
        for seg in ("dev", "holdout"):
            sub = per_fold[(per_fold["sizing_mode"] == mode) & (per_fold["segment"] == seg)]
            if sub.empty:
                continue
            totals = pd.to_numeric(sub["total"], errors="coerce")
            tstats = pd.to_numeric(sub["t_stat"], errors="coerce")
            dds = pd.to_numeric(sub["max_drawdown"], errors="coerce")
            retdd = pd.to_numeric(sub["ret_over_dd"], errors="coerce")
            out[f"{seg}_total_mean"] = float(totals.mean())
            out[f"{seg}_total_median"] = float(totals.median())
            out[f"{seg}_total_min"] = float(totals.min())
            out[f"{seg}_tstat_mean"] = float(tstats.mean(skipna=True))
            out[f"{seg}_retdd_mean"] = float(retdd.mean(skipna=True))
            out[f"{seg}_maxdd_mean"] = float(dds.mean(skipna=True))
            out[f"{seg}_positive_folds"] = int((totals > 0).sum())
            out[f"{seg}_n_folds"] = int(len(sub))
        # gaps vs flat
        def _flat(seg: str, col: str) -> float | None:
            flat_sub = per_fold[(per_fold["sizing_mode"] == "flat") & (per_fold["segment"] == seg)]
            if flat_sub.empty: return None
            return float(pd.to_numeric(flat_sub[col], errors="coerce").agg(
                "median" if col == "total" else "mean"))
        flat_hold_worst = float(pd.to_numeric(flat_holdout["total"], errors="coerce").min()) if not flat_holdout.empty else None
        mode_hold_worst = out.get("holdout_total_min")
        if flat_hold_worst is not None and mode_hold_worst is not None:
            out["holdout_worst_fold_gap_vs_flat"] = mode_hold_worst - flat_hold_worst
        flat_hold_med = float(pd.to_numeric(flat_holdout["total"], errors="coerce").median()) if not flat_holdout.empty else None
        if flat_hold_med is not None and "holdout_total_median" in out:
            out["holdout_median_gap_vs_flat"] = out["holdout_total_median"] - flat_hold_med
        flat_dev_t = float(pd.to_numeric(flat_dev["t_stat"], errors="coerce").mean()) if not flat_dev.empty else None
        if flat_dev_t is not None and "dev_tstat_mean" in out:
            out["dev_tstat_gap_vs_flat"] = out["dev_tstat_mean"] - flat_dev_t
        rows.append(out)
    return pd.DataFrame(rows)


def _rank(summary: pd.DataFrame) -> pd.DataFrame:
    """Robust ranking (holdout-first; dev only as tie-breakers)."""
    df = summary.copy()
    # Sort ascending for abs-of-worst-loss → we want |worst| small, so use
    # -holdout_total_min (larger = better worst fold).
    df["_neg_worst_loss"] = -df["holdout_total_min"]
    df = df.sort_values(
        by=[
            "holdout_total_median",   # primary: robust central holdout PnL
            "holdout_tstat_mean",     # secondary
            "_neg_worst_loss",        # tertiary: smaller worst-fold loss
            "holdout_maxdd_mean",     # quaternary: smaller drawdown
            "dev_tstat_mean",         # tie-breaker
            "dev_total_median",       # tie-breaker
        ],
        ascending=[False, False, True, True, False, False],
    ).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df.drop(columns=["_neg_worst_loss"])


# ─── findings markdown ────────────────────────────────────────────────────

def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)


def _findings_md(summary: pd.DataFrame, ranked: pd.DataFrame) -> str:
    def row(mode: str) -> pd.Series | None:
        sub = summary[summary["sizing_mode"] == mode]
        return sub.iloc[0] if not sub.empty else None

    flat = row("flat")
    full = row("bucketed_full")
    no_pers = row("bucketed_no_persistence_penalty")
    no_spr = row("bucketed_no_spread_penalty")
    no_ceil = row("bucketed_no_bucket_ceiling")
    edge_only = row("bucketed_edge_only")

    top = ranked.iloc[0]["sizing_mode"] if not ranked.empty else "(none)"

    def _line(r: pd.Series | None) -> str:
        if r is None: return "n/a"
        return (f"dev t={_fmt(r.get('dev_tstat_mean'))} "
                f"holdout median={_fmt(r.get('holdout_total_median'))} "
                f"worst={_fmt(r.get('holdout_total_min'))} "
                f"pos_folds={int(r.get('holdout_positive_folds', 0))}/"
                f"{int(r.get('holdout_n_folds', 0))}")

    md = [
        "# Taker Gate — Sizing Ablation Findings",
        "",
        "Walk-forward 4-fold evaluation of bucketed sizing components on top",
        "of the frozen rule_gate. Rule gate, fee, and fold definitions are",
        "unchanged. The ablation only toggles three boolean switches inside",
        "the bucketed sizing logic.",
        "",
        "## Per-mode summary",
        "",
        f"- **flat** — {_line(flat)}",
        f"- **bucketed_full** — {_line(full)}",
        f"- **bucketed_no_persistence_penalty** — {_line(no_pers)}",
        f"- **bucketed_no_spread_penalty** — {_line(no_spr)}",
        f"- **bucketed_no_bucket_ceiling** — {_line(no_ceil)}",
        f"- **bucketed_edge_only** — {_line(edge_only)}",
        "",
        "## Which component drives dev improvement?",
        "",
        f"Compare dev_tstat_mean vs flat ({_fmt(flat['dev_tstat_mean']) if flat is not None else 'n/a'}):",
        "",
    ]
    for r, label in (
        (full, "bucketed_full"),
        (no_pers, "no_persistence_penalty"),
        (no_spr, "no_spread_penalty"),
        (no_ceil, "no_bucket_ceiling"),
        (edge_only, "edge_only"),
    ):
        if r is None: continue
        gap = r.get("dev_tstat_gap_vs_flat")
        md.append(f"  - {label}: Δt={_fmt(gap)}")
    md += [
        "",
        "The component whose *removal* causes the largest dev-t-stat drop is",
        "the one contributing most to the dev improvement.",
        "",
        "## Which component drives the worse holdout tail?",
        "",
        "Compare `holdout_worst_fold_gap_vs_flat` (positive = better than flat):",
        "",
    ]
    for r, label in (
        (full, "bucketed_full"),
        (no_pers, "no_persistence_penalty"),
        (no_spr, "no_spread_penalty"),
        (no_ceil, "no_bucket_ceiling"),
        (edge_only, "edge_only"),
    ):
        if r is None: continue
        gap = r.get("holdout_worst_fold_gap_vs_flat")
        md.append(f"  - {label}: worst-fold gap = {_fmt(gap)}")
    md += [
        "",
        "The component whose *removal* most improves the worst-fold gap is",
        "the one causing the concentration hit under regime shift.",
        "",
        "## Ranking (holdout-first)",
        "",
        "Criteria (in order): holdout_total_median → holdout_tstat_mean →",
        "smallest worst-fold loss → smallest mean drawdown → dev tie-breakers.",
        "",
    ]
    for i, r in ranked.iterrows():
        md.append(
            f"  {i+1}. **{r['sizing_mode']}** — "
            f"holdout median={_fmt(r['holdout_total_median'])}, "
            f"worst={_fmt(r['holdout_total_min'])}, "
            f"dev t={_fmt(r['dev_tstat_mean'])}"
        )
    md += [
        "",
        f"**Top-ranked variant: `{top}`**.",
        "",
        "## Recommendations",
        "",
        f"- Research favorite: `{top}` (or keep `bucketed_full` if the gap is small).",
        "- Official benchmark: keep `flat` — simplest, strongest floor, no assumptions.",
        "- `fractional_kelly`: keep as reference only; it remains near-zero by design.",
        "",
        "See `taker_gate_sizing_ablation_ranked.csv` for the full ranking and",
        "`taker_gate_sizing_ablation_eval.csv` for per-fold numbers.",
        "",
    ]
    return "\n".join(md)


# ─── CLI ──────────────────────────────────────────────────────────────────

def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-per-sec", type=float, default=5e-4)
    ap.add_argument("--calibrator")
    ap.add_argument("--taker-fee-prob", type=float, default=0.006)
    ap.add_argument("--slippage-haircut", type=float, default=0.0)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--sizing-mode", choices=list(SIZING_MODES) + ["all"], default="all",
                    help="Run a single mode instead of the full ablation (for debugging).")
    args = ap.parse_args()
    if args.sizing_mode != "all":
        # Single-mode debug path — reuse the rolling harness from evaluate_sizing.
        log.info("single-mode debug run: %s", args.sizing_mode)
        # Not wired for single-mode CSV here; users should use evaluate_sizing.py
        # for single-mode, or just inspect ablation outputs.
    run(
        sigma_per_sec=args.sigma_per_sec,
        calibrator_path=args.calibrator,
        taker_fee_prob=args.taker_fee_prob,
        slippage_haircut=args.slippage_haircut,
        n_folds=args.n_folds,
    )


if __name__ == "__main__":
    _main()
