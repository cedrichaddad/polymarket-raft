//! Decision router (§12).
//!
//! Given current features, probabilities, and feed health, decide:
//!   * No-trade (§12.3)
//!   * Maker quote one/both sides (§12.2)
//!   * Taker cross (§12.1)
//!
//! The router is deterministic and has no I/O — callers pass it a context and
//! it returns a `Decision`.

use crate::execution::cost_model::{maker_expected_value, AllInCost};
use crate::features::StateFeatures;
use crate::health::FeedHealthSnapshot;
use crate::types::Outcome;

#[derive(Debug, Clone)]
pub struct RouteContext<'a> {
    pub features: &'a StateFeatures,
    pub p_hybrid: f64,
    pub p_market: f64,
    pub all_in_cost: AllInCost,
    pub health: &'a FeedHealthSnapshot,
    pub center_low: f64,
    pub center_high: f64,
    pub min_spread_for_maker: f64,
    /// Seconds remaining below which taker mode is enabled.
    pub taker_window_secs: f64,
    pub expected_adverse_prob: f64,
    pub expected_rebate_prob: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    NoTrade { reason: String },
    MakerQuote {
        outcome: Outcome,
        bid_price: f64,
        ask_price: f64,
    },
    TakerCross {
        outcome: Outcome,
        /// Side to *take* — Buy if we believe p > p_market, Sell otherwise.
        side_is_buy: bool,
        limit_price: f64,
    },
}

pub struct Router;

impl Router {
    pub fn decide(ctx: &RouteContext<'_>) -> Decision {
        // §12.3 — universal no-trade gates.
        if ctx.features.spread_yes <= 0.0 {
            return Decision::NoTrade { reason: "nonpositive_spread".into() };
        }
        if let Some(reason) = health_gate(ctx.health) {
            return Decision::NoTrade { reason };
        }

        let edge = (ctx.p_hybrid - ctx.p_market).abs();
        let threshold = ctx.all_in_cost.total();
        let tte_s = ctx.features.tte_ms as f64 / 1000.0;

        // Taker path — only in late window and with positive all-in edge.
        if tte_s <= ctx.taker_window_secs && edge > threshold {
            let side_is_buy = ctx.p_hybrid > ctx.p_market;
            // Limit price: cross to best opposing side for YES outcome.
            // Without depth we conservatively use the implied midpoint + half-spread.
            let mid = ctx.features.market_prob;
            let half = ctx.features.spread_yes / 2.0;
            let limit_price = if side_is_buy { (mid + half).min(1.0) } else { (mid - half).max(0.0) };
            return Decision::TakerCross {
                outcome: Outcome::Yes,
                side_is_buy,
                limit_price,
            };
        }

        // Maker path — only inside center region with wide-enough spread.
        let p = ctx.p_hybrid;
        if p >= ctx.center_low
            && p <= ctx.center_high
            && ctx.features.spread_yes >= ctx.min_spread_for_maker
        {
            // Maker expected value gate (§12.2).
            let spread_capture = ctx.features.spread_yes / 2.0;
            let ev = maker_expected_value(
                spread_capture,
                ctx.expected_adverse_prob,
                ctx.expected_rebate_prob,
            );
            if ev <= 0.0 {
                return Decision::NoTrade { reason: "maker_ev_nonpositive".into() };
            }
            // Quote one tick inside best bid/ask, clamped to (0,1).
            let bid_price = (p - ctx.features.spread_yes / 2.0 + 0.001).max(0.001);
            let ask_price = (p + ctx.features.spread_yes / 2.0 - 0.001).min(0.999);
            if ask_price <= bid_price {
                return Decision::NoTrade { reason: "inverted_quote".into() };
            }
            return Decision::MakerQuote {
                outcome: Outcome::Yes,
                bid_price,
                ask_price,
            };
        }

        Decision::NoTrade { reason: "no_edge_no_center".into() }
    }
}

fn health_gate(h: &FeedHealthSnapshot) -> Option<String> {
    let binance_stale = h
        .binance_last_update_ms
        .map(|ts| ts == 0 || crate::time_util::now_ms().saturating_sub(ts) > h.binance_stale_ms)
        .unwrap_or(true);
    let chainlink_stale = h
        .chainlink_last_update_ms
        .map(|ts| ts == 0 || crate::time_util::now_ms().saturating_sub(ts) > h.chainlink_stale_ms)
        .unwrap_or(true);
    if binance_stale && chainlink_stale {
        return Some("both_feeds_stale".into());
    }
    if chainlink_stale {
        return Some("chainlink_stale".into());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::StateFeatures;
    use crate::health::FeedHealthSnapshot;
    use crate::time_util::now_ms;

    fn features() -> StateFeatures {
        StateFeatures {
            market_id: "m".into(),
            state_ts_ms: now_ms(),
            tte_ms: 10_000,
            z_open: 0.0,
            z_short: 0.0,
            tte_norm: 0.1,
            spread_yes: 0.02,
            imbalance_yes: 0.0,
            signed_flow_1s: 0.0,
            signed_flow_5s: 0.0,
            rv_10s: 0.001,
            rv_30s: 0.001,
            chainlink_binance_gap: 0.0,
            tick_regime: 1.0,
            market_prob: 0.5,
            chainlink_price: 100_000.0,
            open_ref_price: 100_000.0,
        }
    }

    fn health() -> FeedHealthSnapshot {
        FeedHealthSnapshot {
            rtds_connected: true,
            binance_last_update_ms: Some(now_ms()),
            chainlink_last_update_ms: Some(now_ms()),
            binance_stale_ms: 2000,
            chainlink_stale_ms: 5000,
            parse_errors: 0,
            degraded_reason: None,
        }
    }

    #[test]
    fn taker_triggers_on_big_edge_in_late_window() {
        let f = features();
        let h = health();
        let ctx = RouteContext {
            features: &f,
            p_hybrid: 0.70,
            p_market: 0.50,
            all_in_cost: AllInCost { spread: 0.005, fee: 0.0, slippage: 0.0, adverse: 0.0, model: 0.0 },
            health: &h,
            center_low: 0.4,
            center_high: 0.6,
            min_spread_for_maker: 0.02,
            taker_window_secs: 30.0,
            expected_adverse_prob: 0.0,
            expected_rebate_prob: 0.0,
        };
        let mut f2 = f.clone();
        f2.tte_ms = 20_000; // inside taker window
        let ctx = RouteContext { features: &f2, ..ctx };
        match Router::decide(&ctx) {
            Decision::TakerCross { side_is_buy, .. } => assert!(side_is_buy),
            d => panic!("expected taker, got {d:?}"),
        }
    }

    #[test]
    fn stale_chainlink_blocks_everything() {
        let f = features();
        let h = FeedHealthSnapshot {
            chainlink_last_update_ms: Some(0),
            ..health()
        };
        let ctx = RouteContext {
            features: &f,
            p_hybrid: 0.7,
            p_market: 0.5,
            all_in_cost: AllInCost::default(),
            health: &h,
            center_low: 0.4,
            center_high: 0.6,
            min_spread_for_maker: 0.02,
            taker_window_secs: 30.0,
            expected_adverse_prob: 0.0,
            expected_rebate_prob: 0.0,
        };
        assert!(matches!(Router::decide(&ctx), Decision::NoTrade { .. }));
    }
}
