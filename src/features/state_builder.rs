//! State feature builder (Appendix C).
//!
//! Produces the core state vector from §10.1:
//!   [z_open, z_short, tte_norm, spread_yes, imbalance_yes,
//!    signed_flow_1s, signed_flow_5s, rv_10s, rv_30s,
//!    chainlink_binance_gap, tick_regime, market_prob]
//!
//! Returns `None` when required inputs are missing so the decision layer
//! can refuse to trade (§12.3).

use crate::features::flow::signed_flow_secs;
use crate::features::live_state::{LiveStateStore, MarketMetadata};
use crate::features::vol::{realized_vol_secs, recent_log_return};
use crate::time_util::now_ms;

#[derive(Debug, Clone)]
pub struct StateFeatures {
    pub market_id: String,
    pub state_ts_ms: u64,
    pub tte_ms: u64,
    pub z_open: f64,
    pub z_short: f64,
    pub tte_norm: f64,
    pub spread_yes: f64,
    pub imbalance_yes: f64,
    pub signed_flow_1s: f64,
    pub signed_flow_5s: f64,
    pub rv_10s: f64,
    pub rv_30s: f64,
    pub chainlink_binance_gap: f64,
    pub tick_regime: f64,
    pub market_prob: f64,
    /// Latest Chainlink price (source alignment).
    pub chainlink_price: f64,
    /// Opening reference price for this market window.
    pub open_ref_price: f64,
}

impl StateFeatures {
    /// Returns the state vector in the exact order documented in §10.1.
    pub fn as_vector(&self) -> [f64; 12] {
        [
            self.z_open,
            self.z_short,
            self.tte_norm,
            self.spread_yes,
            self.imbalance_yes,
            self.signed_flow_1s,
            self.signed_flow_5s,
            self.rv_10s,
            self.rv_30s,
            self.chainlink_binance_gap,
            self.tick_regime,
            self.market_prob,
        ]
    }
}

/// Opening reference cache: for each market window we pin the first observed
/// Chainlink print at or after `start_ts_ms`. In research/backtest mode this
/// is canonical; in live mode the Rust service is expected to snapshot
/// `K := chainlink(t=start)` at window start and persist it.
///
/// v1 keeps a simple map; it's on the caller to fill it.
pub type OpenRefResolver<'a> = &'a dyn Fn(&MarketMetadata) -> Option<f64>;

pub fn build_state_features(
    store: &LiveStateStore,
    market_id: &str,
    open_ref: OpenRefResolver,
    horizon_seconds: f64,
) -> Option<StateFeatures> {
    let meta = store.metadata.get(market_id)?.clone();
    let mkt = store.markets.get(market_id)?.clone();
    let refs = store.reference.read().clone();
    let chainlink = refs.chainlink_price?;
    let binance = refs.binance_price;

    let ts = now_ms();
    let tte_ms = meta.end_ts_ms.saturating_sub(ts);
    let tte_seconds = (tte_ms as f64) / 1000.0;
    let tte_norm = (tte_seconds / horizon_seconds).clamp(0.0, 1.0);

    let spread_yes = match (mkt.best_yes_bid, mkt.best_yes_ask) {
        (Some(b), Some(a)) => a - b,
        _ => return None,
    };
    // L1 imbalance proxy: (bid - ask) / (bid + ask) on yes. Not perfect but
    // sufficient for v1; refine once book depth wiring lands.
    let imbalance_yes = match (mkt.best_yes_bid, mkt.best_yes_ask) {
        (Some(b), Some(a)) if b + a > 0.0 => (b - a) / (b + a),
        _ => 0.0,
    };

    let open_ref_price = open_ref(&meta)?;
    if open_ref_price <= 0.0 || chainlink <= 0.0 {
        return None;
    }

    let chainlink_samples = store.chainlink_samples();
    let rv_10s = realized_vol_secs(&chainlink_samples, ts, 10).unwrap_or(0.0);
    let rv_30s = realized_vol_secs(&chainlink_samples, ts, 30).unwrap_or(0.0);
    let ret_short = recent_log_return(&chainlink_samples, ts, 5).unwrap_or(0.0);
    let ret_5s_to_open = (chainlink / open_ref_price).ln();

    // Normalize z_open by rv_30s * sqrt(tte_seconds). If vol is 0 we fall back
    // to a small epsilon to avoid NaN; the decision layer should treat very
    // small denominators as a no-trade gate.
    let denom_open = (rv_30s * (tte_seconds + 1.0).sqrt()).max(1e-6);
    let z_open = ret_5s_to_open / denom_open;
    let denom_short = rv_10s.max(1e-6);
    let z_short = ret_short / denom_short;

    let trades = store.recent_trades(market_id);
    let signed_flow_1s = signed_flow_secs(&trades, ts, 1);
    let signed_flow_5s = signed_flow_secs(&trades, ts, 5);

    let chainlink_binance_gap = binance.map(|b| chainlink - b).unwrap_or(0.0);

    // Tick regime: 1.0 for center (|mid-0.5| <= 0.1), 0.0 for wing.
    let market_prob = mkt.mid_yes().unwrap_or(0.5).clamp(0.0, 1.0);
    let tick_regime = if (market_prob - 0.5).abs() <= 0.1 { 1.0 } else { 0.0 };

    Some(StateFeatures {
        market_id: market_id.to_string(),
        state_ts_ms: ts,
        tte_ms,
        z_open,
        z_short,
        tte_norm,
        spread_yes,
        imbalance_yes,
        signed_flow_1s,
        signed_flow_5s,
        rv_10s,
        rv_30s,
        chainlink_binance_gap,
        tick_regime,
        market_prob,
        chainlink_price: chainlink,
        open_ref_price,
    })
}
