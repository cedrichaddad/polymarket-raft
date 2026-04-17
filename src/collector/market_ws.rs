//! Polymarket market WebSocket collector (§7.5, §7.7.2).
//!
//! v1 responsibilities:
//!   * subscribe to the active market token set
//!   * parse channel messages into `MarketBookEvent` / `MarketTradeEvent`
//!   * publish normalized events to the bus
//!
//! The Polymarket `/market` channel exposes `book` / `price_change` / `last_trade_price`
//! message types over a JSON envelope. We parse defensively: unknown events are
//! counted and skipped rather than aborting the connection.

use crate::collector::backoff::Backoff;
use crate::config::MarketWsConfig;
use crate::event_bus::EventSender;
use crate::time_util::now_ms;
use crate::types::{ExternalEvent, MarketBookEvent, MarketTradeEvent, Side};

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info, warn};

pub struct MarketWsClient {
    cfg: MarketWsConfig,
    bus: EventSender,
    /// Asset IDs (CLOB token IDs) to subscribe to.
    assets: Arc<RwLock<Vec<String>>>,
}

impl MarketWsClient {
    pub fn new(cfg: MarketWsConfig, bus: EventSender) -> Self {
        Self {
            cfg,
            bus,
            assets: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Replace the subscribed asset set. On next reconnect the new set is used;
    /// we don't currently support incremental subscribe deltas.
    pub async fn set_assets(&self, assets: Vec<String>) {
        *self.assets.write().await = assets;
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        use crate::config::BackoffConfig;
        let backoff_cfg = BackoffConfig {
            initial_backoff_ms: self.cfg.reconnect_initial_backoff_ms,
            max_backoff_ms: self.cfg.reconnect_max_backoff_ms,
            jitter: true,
        };
        let mut b = Backoff::new(&backoff_cfg);
        loop {
            match self.run_once().await {
                Ok(()) => warn!(target: "mktws", "closed cleanly, reconnecting"),
                Err(e) => error!(target: "mktws", error = %e, "market ws error"),
            }
            tokio::time::sleep(b.next_delay()).await;
        }
    }

    async fn run_once(&self) -> Result<()> {
        let assets = self.assets.read().await.clone();
        if assets.is_empty() {
            // Nothing to subscribe to yet; wait and retry without burning a reconnect.
            debug!(target: "mktws", "no assets configured, idling");
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
            return Ok(());
        }

        info!(target: "mktws", url = %self.cfg.url, n = assets.len(), "connecting");
        let (ws, _) = connect_async(&self.cfg.url).await.context("mkt connect_async")?;
        let (mut write, mut read) = ws.split();

        // Polymarket market subscription shape: {"type":"market","assets_ids":[...]}.
        let subscribe = json!({
            "type": "market",
            "assets_ids": assets,
        });
        write
            .send(Message::Text(subscribe.to_string()))
            .await
            .context("market subscribe")?;
        info!(target: "mktws", "subscribed to {} assets", assets.len());

        while let Some(msg) = read.next().await {
            match msg {
                Ok(Message::Text(t)) => {
                    if let Err(e) = self.handle_text(&t) {
                        warn!(target: "mktws", error = %e, "parse error");
                    }
                }
                Ok(Message::Ping(p)) => {
                    let _ = write.send(Message::Pong(p)).await;
                }
                Ok(Message::Close(_)) => return Ok(()),
                Ok(_) => {}
                Err(e) => return Err(anyhow!("mkt ws read error: {e}")),
            }
        }
        Ok(())
    }

    fn handle_text(&self, text: &str) -> Result<()> {
        // Responses can be either a single object or an array of events.
        let v: Value = serde_json::from_str(text).context("decode market json")?;
        match v {
            Value::Array(arr) => {
                for item in arr {
                    self.handle_value(item);
                }
            }
            other => self.handle_value(other),
        }
        Ok(())
    }

    fn handle_value(&self, v: Value) {
        let Some(event_type) = v.get("event_type").and_then(|x| x.as_str()) else {
            return;
        };
        let market_id = v
            .get("market")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let asset_id = v
            .get("asset_id")
            .and_then(|x| x.as_str())
            .unwrap_or_default()
            .to_string();
        let ts_exchange_ms = v
            .get("timestamp")
            .and_then(|x| x.as_str().and_then(|s| s.parse::<u64>().ok()).or_else(|| x.as_u64()));

        match event_type {
            "book" | "price_change" => {
                let (best_bid, best_ask) = extract_best_levels(&v);
                let spread = match (best_bid, best_ask) {
                    (Some(b), Some(a)) => Some(a - b),
                    _ => None,
                };
                let ev = MarketBookEvent {
                    ts_exchange_ms,
                    ts_recv_local_ms: now_ms(),
                    market_id,
                    asset_id,
                    event_type: event_type.to_string(),
                    best_bid,
                    best_ask,
                    spread,
                    book_json: v.to_string(),
                };
                let _ = self.bus.send(ExternalEvent::MarketBook(ev));
            }
            "last_trade_price" | "trade" => {
                let price = v.get("price").and_then(json_to_f64).unwrap_or(0.0);
                let size = v.get("size").and_then(json_to_f64).unwrap_or(0.0);
                let side_aggressor = v.get("side").and_then(|x| x.as_str()).and_then(parse_side);
                let trade_id = v.get("trade_id").and_then(|x| x.as_str()).map(String::from);
                let ev = MarketTradeEvent {
                    ts_exchange_ms,
                    ts_recv_local_ms: now_ms(),
                    market_id,
                    asset_id,
                    price,
                    size,
                    side_aggressor,
                    trade_id,
                };
                let _ = self.bus.send(ExternalEvent::MarketTrade(ev));
            }
            other => {
                debug!(target: "mktws", event = other, "ignored event type");
            }
        }
    }
}

fn parse_side(s: &str) -> Option<Side> {
    match s.to_ascii_uppercase().as_str() {
        "BUY" => Some(Side::Buy),
        "SELL" => Some(Side::Sell),
        _ => None,
    }
}

fn json_to_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

/// Extract best bid/ask from a book or price_change payload.
/// Polymarket books expose `bids`/`asks` arrays of {price, size} entries; best
/// bid is the highest price on the bid side and best ask is the lowest on ask.
fn extract_best_levels(v: &Value) -> (Option<f64>, Option<f64>) {
    let best_bid = v
        .get("bids")
        .and_then(|b| b.as_array())
        .and_then(|arr| {
            arr.iter()
                .filter_map(|e| e.get("price").and_then(json_to_f64))
                .fold(None, |acc: Option<f64>, p| {
                    Some(acc.map_or(p, |x| x.max(p)))
                })
        });
    let best_ask = v
        .get("asks")
        .and_then(|b| b.as_array())
        .and_then(|arr| {
            arr.iter()
                .filter_map(|e| e.get("price").and_then(json_to_f64))
                .fold(None, |acc: Option<f64>, p| {
                    Some(acc.map_or(p, |x| x.min(p)))
                })
        });
    // price_change may provide a single `price`/`side` instead of levels.
    if best_bid.is_none() && best_ask.is_none() {
        if let (Some(price), Some(side)) = (
            v.get("price").and_then(json_to_f64),
            v.get("side").and_then(|x| x.as_str()),
        ) {
            match side.to_ascii_uppercase().as_str() {
                "BUY" => return (Some(price), None),
                "SELL" => return (None, Some(price)),
                _ => {}
            }
        }
    }
    (best_bid, best_ask)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_payload_yields_best_bid_ask() {
        let v = json!({
            "event_type": "book",
            "market": "m1",
            "asset_id": "a1",
            "bids": [{"price": "0.48", "size": "10"}, {"price": "0.49", "size": "5"}],
            "asks": [{"price": "0.52", "size": "8"}, {"price": "0.51", "size": "3"}],
            "timestamp": "1700000000000"
        });
        let (b, a) = extract_best_levels(&v);
        assert!((b.unwrap() - 0.49).abs() < 1e-9);
        assert!((a.unwrap() - 0.51).abs() < 1e-9);
    }

    #[test]
    fn price_change_reports_single_side() {
        let v = json!({"event_type": "price_change", "price": "0.55", "side": "BUY"});
        let (b, a) = extract_best_levels(&v);
        assert_eq!(b, Some(0.55));
        assert_eq!(a, None);
    }
}
