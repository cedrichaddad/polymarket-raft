//! Broker abstraction.
//!
//! The execution layer produces `OrderIntent`s; a `Broker` is responsible for
//! actually placing them (paper: in-memory; live: Polymarket CLOB signed
//! order). v1 ships with `PaperBroker` only — live brokerage is a later
//! milestone.

use crate::time_util::now_ms;
use crate::types::{FillEvent, OrderAckEvent, Side};
use parking_lot::Mutex;
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct OrderIntent {
    pub market_id: String,
    pub asset_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub post_only: bool,
    pub expiration_ts_ms: Option<u64>,
    /// Free-form reason the router chose this order (for logs / audits).
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum BrokerEvent {
    Ack(OrderAckEvent),
    Fill(FillEvent),
}

pub trait Broker: Send + Sync {
    fn place(&self, intent: OrderIntent) -> BrokerEvent;
    fn cancel(&self, exchange_order_id: &str);
}

/// In-memory paper broker. It fills any marketable order at the submitted
/// price instantly and rests post-only orders. Real fill-simulation lives in
/// the Python backtester (§14); this is a minimal stand-in so paper mode can
/// feed the rest of the system.
pub struct PaperBroker {
    open: Mutex<Vec<OrderAckEvent>>,
    fees: PaperFees,
}

#[derive(Debug, Clone, Copy)]
pub struct PaperFees {
    pub taker_fee_prob: f64,
    pub maker_rebate_prob: f64,
}

impl Default for PaperFees {
    fn default() -> Self {
        Self {
            taker_fee_prob: 0.0,
            maker_rebate_prob: 0.0,
        }
    }
}

impl PaperBroker {
    pub fn new(fees: PaperFees) -> Arc<Self> {
        Arc::new(Self {
            open: Mutex::new(Vec::new()),
            fees,
        })
    }

    pub fn open_orders(&self) -> Vec<OrderAckEvent> {
        self.open.lock().clone()
    }
}

impl Broker for PaperBroker {
    fn place(&self, intent: OrderIntent) -> BrokerEvent {
        let local_id = Uuid::new_v4().to_string();
        let exchange_order_id = Some(Uuid::new_v4().to_string());
        let status = if intent.post_only { "ACCEPTED_RESTING" } else { "FILLED" };
        let ack = OrderAckEvent {
            ts_recv_local_ms: now_ms(),
            local_order_id: local_id.clone(),
            exchange_order_id: exchange_order_id.clone(),
            market_id: intent.market_id.clone(),
            asset_id: intent.asset_id.clone(),
            side: intent.side,
            price: intent.price,
            size: intent.size,
            order_type: if intent.post_only { "POST_ONLY" } else { "FAK" }.to_string(),
            post_only: intent.post_only,
            expiration_ts_ms: intent.expiration_ts_ms,
            status: status.into(),
        };
        if intent.post_only {
            self.open.lock().push(ack.clone());
            BrokerEvent::Ack(ack)
        } else {
            let fill = FillEvent {
                fill_ts_ms: now_ms(),
                exchange_order_id: exchange_order_id.unwrap_or_default(),
                market_id: intent.market_id,
                asset_id: intent.asset_id,
                fill_price: intent.price,
                fill_size: intent.size,
                maker_or_taker: "taker".into(),
                estimated_fee_usdc: intent.size * intent.price * self.fees.taker_fee_prob,
                estimated_rebate_usdc: 0.0,
            };
            BrokerEvent::Fill(fill)
        }
    }

    fn cancel(&self, exchange_order_id: &str) {
        self.open
            .lock()
            .retain(|o| o.exchange_order_id.as_deref() != Some(exchange_order_id));
    }
}
