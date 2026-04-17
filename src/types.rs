//! Normalized event types emitted by collectors and consumed downstream.
//! See §7.8.

use serde::{Deserialize, Serialize};

/// Source of a crypto price update.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RtdsTopic {
    Binance,
    Chainlink,
}

impl RtdsTopic {
    pub fn as_str(self) -> &'static str {
        match self {
            RtdsTopic::Binance => "binance",
            RtdsTopic::Chainlink => "chainlink",
        }
    }
}

/// Normalized RTDS crypto price update (§7.4.7).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtdsCryptoUpdate {
    pub topic: RtdsTopic,
    pub symbol: String,
    pub ts_server_ms: u64,
    pub ts_payload_ms: u64,
    pub ts_recv_local_ms: u64,
    pub value: f64,
}

/// Side of a market order / trade aggressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Buy,
    Sell,
}

/// Outcome token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Yes,
    No,
}

/// Top-of-book snapshot for a single market/asset pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketBookEvent {
    pub ts_exchange_ms: Option<u64>,
    pub ts_recv_local_ms: u64,
    pub market_id: String,
    pub asset_id: String,
    pub event_type: String,
    pub best_bid: Option<f64>,
    pub best_ask: Option<f64>,
    pub spread: Option<f64>,
    /// Raw event body for archival (kept as JSON string so we can persist to Parquet).
    pub book_json: String,
}

/// Trade print.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTradeEvent {
    pub ts_exchange_ms: Option<u64>,
    pub ts_recv_local_ms: u64,
    pub market_id: String,
    pub asset_id: String,
    pub price: f64,
    pub size: f64,
    pub side_aggressor: Option<Side>,
    pub trade_id: Option<String>,
}

/// Market metadata (§9.1, §7.6).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketMetaEvent {
    pub market_id: String,
    pub asset_yes_id: String,
    pub asset_no_id: String,
    pub window_type: String, // "btc_5m" | "btc_15m"
    pub start_ts_ms: u64,
    pub end_ts_ms: u64,
    pub resolution_source: String,
    pub fees_enabled: bool,
    pub fee_schedule_json: String,
    pub tick_size: Option<f64>,
}

/// Local order bookkeeping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderAckEvent {
    pub ts_recv_local_ms: u64,
    pub local_order_id: String,
    pub exchange_order_id: Option<String>,
    pub market_id: String,
    pub asset_id: String,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub order_type: String,
    pub post_only: bool,
    pub expiration_ts_ms: Option<u64>,
    pub status: String,
}

/// Fill event (§9.1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    pub fill_ts_ms: u64,
    pub exchange_order_id: String,
    pub market_id: String,
    pub asset_id: String,
    pub fill_price: f64,
    pub fill_size: f64,
    pub maker_or_taker: String,
    pub estimated_fee_usdc: f64,
    pub estimated_rebate_usdc: f64,
}

/// Union of normalized internal events (§7.8).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExternalEvent {
    RtdsCrypto(RtdsCryptoUpdate),
    MarketBook(MarketBookEvent),
    MarketTrade(MarketTradeEvent),
    MarketMeta(MarketMetaEvent),
    OrderAck(OrderAckEvent),
    Fill(FillEvent),
}

impl ExternalEvent {
    pub fn kind(&self) -> &'static str {
        match self {
            ExternalEvent::RtdsCrypto(_) => "rtds_crypto",
            ExternalEvent::MarketBook(_) => "market_book",
            ExternalEvent::MarketTrade(_) => "market_trade",
            ExternalEvent::MarketMeta(_) => "market_meta",
            ExternalEvent::OrderAck(_) => "order_ack",
            ExternalEvent::Fill(_) => "fill",
        }
    }
}

/// In-memory live reference prices (latest values).
#[derive(Debug, Clone, Default)]
pub struct LiveReferenceState {
    pub chainlink_price: Option<f64>,
    pub chainlink_ts_ms: Option<u64>,
    pub binance_price: Option<f64>,
    pub binance_ts_ms: Option<u64>,
}

/// In-memory live market state per market.
#[derive(Debug, Clone, Default)]
pub struct LiveMarketState {
    pub market_id: String,
    pub best_yes_bid: Option<f64>,
    pub best_yes_ask: Option<f64>,
    pub best_no_bid: Option<f64>,
    pub best_no_ask: Option<f64>,
    pub last_trade_price: Option<f64>,
    pub last_trade_ts_ms: Option<u64>,
    pub last_book_update_ts_ms: Option<u64>,
}

impl LiveMarketState {
    pub fn mid_yes(&self) -> Option<f64> {
        match (self.best_yes_bid, self.best_yes_ask) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }
    pub fn spread_yes(&self) -> Option<f64> {
        match (self.best_yes_bid, self.best_yes_ask) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }
}
