//! In-memory live state store.
//!
//! A `LiveStateStore` ingests `ExternalEvent`s and maintains:
//!   * `LiveReferenceState` — latest Chainlink + Binance values
//!   * `LiveMarketState` — per-market top of book / last trade
//!   * `MarketMetadata` — per-market static metadata (fees, window bounds)
//!   * Rolling trade/flow/return buffers for the vol and flow modules
//!
//! All state lives behind `RwLock`s so the feature builder, execution layer,
//! and Parquet sink can all read from it concurrently.

use crate::types::{
    ExternalEvent, LiveMarketState, LiveReferenceState, MarketMetaEvent, Side,
};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone, Default)]
pub struct MarketMetadata {
    pub market_id: String,
    pub asset_yes_id: String,
    pub asset_no_id: String,
    pub window_type: String,
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    pub fees_enabled: bool,
    pub tick_size: Option<f64>,
}

/// Recent trade print (signed size, price).
#[derive(Debug, Clone, Copy)]
pub struct TradeSample {
    pub ts_ms: u64,
    pub signed_size: f64,
    pub price: f64,
}

/// Recent reference-price sample (for vol / returns).
#[derive(Debug, Clone, Copy)]
pub struct RefPriceSample {
    pub ts_ms: u64,
    pub value: f64,
}

pub struct LiveStateStore {
    pub reference: RwLock<LiveReferenceState>,
    pub markets: DashMap<String, LiveMarketState>,
    pub metadata: DashMap<String, MarketMetadata>,
    /// asset_id -> which market it belongs to (and whether yes/no).
    pub asset_index: DashMap<String, AssetRef>,
    /// Recent signed-trade prints per market (bounded).
    pub trade_window: DashMap<String, VecDeque<TradeSample>>,
    /// Recent Chainlink prints (for realized vol / returns).
    pub chainlink_window: RwLock<VecDeque<RefPriceSample>>,
    window_cap_trades: usize,
    window_cap_prices: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetRef {
    Yes,
    No,
}

impl Default for LiveStateStore {
    fn default() -> Self {
        Self::new()
    }
}

impl LiveStateStore {
    pub fn new() -> Self {
        Self {
            reference: RwLock::new(LiveReferenceState::default()),
            markets: DashMap::new(),
            metadata: DashMap::new(),
            asset_index: DashMap::new(),
            trade_window: DashMap::new(),
            chainlink_window: RwLock::new(VecDeque::with_capacity(256)),
            window_cap_trades: 256,
            window_cap_prices: 256,
        }
    }

    pub fn apply(&self, ev: &ExternalEvent) {
        match ev {
            ExternalEvent::RtdsCrypto(u) => {
                let mut r = self.reference.write();
                match u.topic {
                    crate::types::RtdsTopic::Binance => {
                        r.binance_price = Some(u.value);
                        r.binance_ts_ms = Some(u.ts_payload_ms.max(u.ts_recv_local_ms));
                    }
                    crate::types::RtdsTopic::Chainlink => {
                        r.chainlink_price = Some(u.value);
                        r.chainlink_ts_ms = Some(u.ts_payload_ms.max(u.ts_recv_local_ms));
                        drop(r);
                        let mut w = self.chainlink_window.write();
                        if w.len() >= self.window_cap_prices {
                            w.pop_front();
                        }
                        w.push_back(RefPriceSample {
                            ts_ms: u.ts_payload_ms.max(u.ts_recv_local_ms),
                            value: u.value,
                        });
                    }
                }
            }
            ExternalEvent::MarketBook(b) => {
                let mut state = self
                    .markets
                    .entry(b.market_id.clone())
                    .or_insert_with(|| LiveMarketState {
                        market_id: b.market_id.clone(),
                        ..Default::default()
                    });
                let is_yes = self
                    .asset_index
                    .get(&b.asset_id)
                    .map(|x| *x == AssetRef::Yes)
                    .unwrap_or(true); // default assume yes if unknown
                if is_yes {
                    state.best_yes_bid = b.best_bid;
                    state.best_yes_ask = b.best_ask;
                } else {
                    state.best_no_bid = b.best_bid;
                    state.best_no_ask = b.best_ask;
                }
                state.last_book_update_ts_ms = Some(b.ts_recv_local_ms);
            }
            ExternalEvent::MarketTrade(t) => {
                let signed = match t.side_aggressor {
                    Some(Side::Buy) => t.size,
                    Some(Side::Sell) => -t.size,
                    None => 0.0,
                };
                let mut w = self.trade_window.entry(t.market_id.clone()).or_default();
                if w.len() >= self.window_cap_trades {
                    w.pop_front();
                }
                w.push_back(TradeSample {
                    ts_ms: t.ts_recv_local_ms,
                    signed_size: signed,
                    price: t.price,
                });
                drop(w);
                let mut state = self
                    .markets
                    .entry(t.market_id.clone())
                    .or_insert_with(|| LiveMarketState {
                        market_id: t.market_id.clone(),
                        ..Default::default()
                    });
                state.last_trade_price = Some(t.price);
                state.last_trade_ts_ms = Some(t.ts_recv_local_ms);
            }
            ExternalEvent::MarketMeta(m) => {
                self.upsert_metadata(m.clone());
            }
            ExternalEvent::OrderAck(_) | ExternalEvent::Fill(_) => {
                // Order/fill state lives elsewhere (execution router).
            }
        }
    }

    fn upsert_metadata(&self, m: MarketMetaEvent) {
        self.asset_index.insert(m.asset_yes_id.clone(), AssetRef::Yes);
        self.asset_index.insert(m.asset_no_id.clone(), AssetRef::No);
        self.metadata.insert(
            m.market_id.clone(),
            MarketMetadata {
                market_id: m.market_id,
                asset_yes_id: m.asset_yes_id,
                asset_no_id: m.asset_no_id,
                window_type: m.window_type,
                start_ts_ms: m.start_ts_ms,
                end_ts_ms: m.end_ts_ms,
                fees_enabled: m.fees_enabled,
                tick_size: m.tick_size,
            },
        );
    }

    pub fn chainlink_samples(&self) -> Vec<RefPriceSample> {
        self.chainlink_window.read().iter().copied().collect()
    }

    pub fn recent_trades(&self, market_id: &str) -> Vec<TradeSample> {
        self.trade_window
            .get(market_id)
            .map(|w| w.iter().copied().collect())
            .unwrap_or_default()
    }
}

/// Public helper so the bus-consumer loop can use an `Arc<LiveStateStore>`.
pub fn apply_to(store: &Arc<LiveStateStore>, ev: &ExternalEvent) {
    store.apply(ev);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{RtdsCryptoUpdate, RtdsTopic};

    #[test]
    fn reference_price_is_updated() {
        let s = LiveStateStore::new();
        s.apply(&ExternalEvent::RtdsCrypto(RtdsCryptoUpdate {
            topic: RtdsTopic::Chainlink,
            symbol: "btc/usd".into(),
            ts_server_ms: 1,
            ts_payload_ms: 1,
            ts_recv_local_ms: 2,
            value: 100_500.0,
        }));
        let r = s.reference.read();
        assert_eq!(r.chainlink_price, Some(100_500.0));
    }
}
