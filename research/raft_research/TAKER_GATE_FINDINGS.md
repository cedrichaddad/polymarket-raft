# Taker Gate — Findings

Scope: late-window, time-to-expiry–aware taker entry gate for Polymarket BTC
minute contracts. Compares three modes on the full in-sample set and on a
market-level holdout (30 % of newest markets reserved before training).

## Headline numbers

### Full in-sample (no holdout)
| Mode       | n  | unique mkts | mean net PnL | t-stat | total | hit rate |
|------------|---:|------------:|-------------:|-------:|------:|---------:|
| baseline   | 76 | 76          | 0.064        | 0.87   |  4.87 | 0.500 |
| rule_gate  | 54 | 54          | 0.117        | 1.39   |  6.31 | 0.574 |
| model_gate | 79 | 79          | 0.223        | 3.30   | 17.61 | 0.671 |

### Honest holdout (train/calibrate on 70 % oldest markets, evaluate on 30 % newest)
| Mode       | n  | mean net PnL | t-stat | total | hit rate |
|------------|---:|-------------:|-------:|------:|---------:|
| baseline   | 22 | -0.004       | -0.03  | -0.09 | 0.409 |
| rule_gate  | 15 |  0.181       |  0.93  |  2.72 | 0.600 |
| model_gate | 18 |  0.009       |  0.06  |  0.16 | 0.722 |

## Findings

1. **Late-window gating helps in-sample**, but the in-sample gains do not
   fully survive an honest market-level holdout. The 3.3σ in-sample t-stat of
   the learned gate collapses to 0.06 out-of-sample — classic overfit with
   only ~1 000 training candidates and 79 markets.

2. **The rule-based gate is the most robust variant.** On holdout it keeps a
   positive mean (0.181) and 60 % hit rate with only a modest 0.93 t-stat on
   15 trades — the only mode with directional edge that survives.

3. **The `10_15` bucket is the workhorse.** Per-bucket diagnostics on the
   in-sample rule_gate trades: `10_15` = 51 trades / mean 0.18, and in the
   sweep the top model_gate rows are also dominated by `10_15`. The `15_30`
   bucket is disabled by default and should remain so — its mean is negative
   or near-zero everywhere we looked. The `0_2` bucket works when any trade
   lands there but the sample is tiny (1–3 trades) and execution risk is
   highest there, so keep it disabled by default.

4. **Persistence is a weak filter on its own.** Conditional PnL with vs
   without `edge_persist_2s` is almost identical (mean 0.22 vs 0.30 in
   diagnostics). It is useful mainly as a tie-breaker: it lets us widen the
   entry window without letting one-bar spikes in.

5. **Fees matter at the margin.** Across the sweep, the top rule/model gate
   configurations keep positive t-stat up to a 1.2 % taker fee; baseline
   crosses zero before that. Gated modes are fee-robust precisely because
   they trade fewer but higher-confidence bars.

6. **Maker is not helped by the same bucketing.** Maker fills concentrate
   far from expiry (mean tte ≈ 314 s); the small `10_15` tail shows mean
   0.13. Maker is a different problem — bucketing it the same way does not
   add signal.

## Recommended default config

- `mode = rule_gate`
- `max_entry_window_secs = 15`
- `require_persistence = True`, `persist_bars = 2`
- Enabled buckets: `10_15: 0.03`, `5_10: 0.025`, `2_5: 0.02`
- Disabled: `15_30`, `0_2`
- Quote quality: `max_spread = 0.05`, `max_chainlink_binance_gap_abs = 500` USD

Prefer `rule_gate` over `model_gate` until we have at least 3× more labeled
markets. The rule gate generalizes; the logistic model as-trained does not.

## Caveats

- Holdout has only ~25 markets. Effect sizes above 1σ should be treated as
  directional hints, not proof.
- The "net PnL" here assumes a conservative full-half-spread fill and the
  configured taker fee; slippage beyond that is not modeled.
- Calibrator is LOMO; compare.py overlays retrieval but that path is not
  exercised in these runs.

## Artifacts

- Sweeps: `data/derived/sweeps/taker_gate_sweep.csv`
- Holdout: `data/derived/taker_gate_holdout_summary.json`,
  `data/derived/taker_gate_holdout_trades.parquet`
- Diagnostics: `data/derived/taker_gate_{bucket_stats,edge_deciles,persistence_stats}.csv`
- Maker companion: `data/derived/maker_bucket_stats.csv`
- Models: `data/derived/taker_gate_model.json`,
  `data/derived/taker_gate_model_metrics.json`
- Baseline snapshot: `data/derived/taker_gate_baseline.json`
