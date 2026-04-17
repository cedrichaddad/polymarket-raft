//! Maker quote lifecycle manager (§13.2).
//!
//! Tracks one resting bid/ask pair per (market, outcome). When the router
//! emits a `MakerQuote` decision, `QuoteManager::apply` decides whether to:
//!   * hold the existing quote
//!   * reprice (cancel + replace)
//!   * cancel only (if router says no-trade)
//!
//! Cancel triggers (§13.2): target moved >= 1 tick, imbalance spike, stale
//! quote age, spread collapse, tick regime change. Only the price-based
//! triggers are implemented here; upstream code can force a cancel via
//! `force_cancel_all`.

use crate::execution::broker::{Broker, OrderIntent};
use crate::execution::router::Decision;
use crate::time_util::now_ms;
use crate::types::{Outcome, Side};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct RestingQuote {
    pub exchange_order_id: Option<String>,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub placed_ms: u64,
}

pub struct QuoteManager {
    broker: Arc<dyn Broker>,
    state: Mutex<HashMap<(String, Outcome), (Option<RestingQuote>, Option<RestingQuote>)>>,
    /// Max age before a quote is considered stale and force-reprised.
    pub max_quote_age_ms: u64,
    /// Minimum tick to treat as "moved enough to reprise".
    pub reprise_tick: f64,
    pub default_size: f64,
}

impl QuoteManager {
    pub fn new(broker: Arc<dyn Broker>) -> Self {
        Self {
            broker,
            state: Mutex::new(HashMap::new()),
            max_quote_age_ms: 2_000,
            reprise_tick: 0.01,
            default_size: 10.0,
        }
    }

    pub fn apply(&self, market_id: &str, asset_id: &str, decision: &Decision) {
        match decision {
            Decision::MakerQuote { outcome, bid_price, ask_price } => {
                self.apply_maker(market_id, asset_id, *outcome, *bid_price, *ask_price)
            }
            Decision::NoTrade { .. } | Decision::TakerCross { .. } => {
                self.cancel_market(market_id);
            }
        }
    }

    fn apply_maker(
        &self,
        market_id: &str,
        asset_id: &str,
        outcome: Outcome,
        bid_price: f64,
        ask_price: f64,
    ) {
        let key = (market_id.to_string(), outcome);
        let mut state = self.state.lock();
        let (bid_slot, ask_slot) = state.entry(key).or_insert((None, None));
        self.maybe_replace(
            market_id,
            asset_id,
            Side::Buy,
            bid_price,
            bid_slot,
        );
        self.maybe_replace(
            market_id,
            asset_id,
            Side::Sell,
            ask_price,
            ask_slot,
        );
    }

    fn maybe_replace(
        &self,
        market_id: &str,
        asset_id: &str,
        side: Side,
        target_price: f64,
        slot: &mut Option<RestingQuote>,
    ) {
        let now = now_ms();
        let should_replace = match slot {
            Some(rq) => {
                let age = now.saturating_sub(rq.placed_ms);
                let moved = (rq.price - target_price).abs() >= self.reprise_tick;
                moved || age >= self.max_quote_age_ms
            }
            None => true,
        };
        if !should_replace {
            return;
        }
        if let Some(rq) = slot.take() {
            if let Some(id) = rq.exchange_order_id {
                self.broker.cancel(&id);
            }
        }
        let intent = OrderIntent {
            market_id: market_id.to_string(),
            asset_id: asset_id.to_string(),
            side,
            price: target_price,
            size: self.default_size,
            post_only: true,
            expiration_ts_ms: None,
            reason: "maker_quote".into(),
        };
        let evt = self.broker.place(intent);
        if let crate::execution::broker::BrokerEvent::Ack(ack) = evt {
            *slot = Some(RestingQuote {
                exchange_order_id: ack.exchange_order_id,
                side,
                price: target_price,
                size: self.default_size,
                placed_ms: now,
            });
        }
    }

    pub fn cancel_market(&self, market_id: &str) {
        let mut state = self.state.lock();
        let keys: Vec<_> = state.keys().filter(|(m, _)| m == market_id).cloned().collect();
        for k in keys {
            if let Some((bid, ask)) = state.get_mut(&k) {
                cancel_slot(self.broker.as_ref(), bid);
                cancel_slot(self.broker.as_ref(), ask);
            }
        }
    }

    pub fn force_cancel_all(&self) {
        let mut state = self.state.lock();
        for (_, (bid, ask)) in state.iter_mut() {
            cancel_slot(self.broker.as_ref(), bid);
            cancel_slot(self.broker.as_ref(), ask);
        }
    }
}

fn cancel_slot(broker: &dyn Broker, slot: &mut Option<RestingQuote>) {
    if let Some(rq) = slot.take() {
        if let Some(id) = rq.exchange_order_id {
            broker.cancel(&id);
        }
    }
}
