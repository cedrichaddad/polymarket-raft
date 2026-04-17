//! RTDS WebSocket client (§7.4).
//!
//! Responsibilities (§7.7.1):
//!   * socket connect / reconnect
//!   * subscribe / unsubscribe envelopes
//!   * ping loop (literal `PING` every 5s, §7.4.5)
//!   * parse RTDS envelopes
//!   * publish normalized `RtdsCryptoUpdate` to the event bus
//!
//! Invariants (§7.4.9): dedupe by `(topic, symbol, ts_payload_ms, value)`, record
//! both receive and payload timestamps, tolerate out-of-order arrival.

use crate::collector::backoff::Backoff;
use crate::config::{RtdsConfig, RtdsSubscriptionConfig};
use crate::event_bus::EventSender;
use crate::health::HealthMonitor;
use crate::time_util::now_ms;
use crate::types::{ExternalEvent, RtdsCryptoUpdate, RtdsTopic};

use anyhow::{anyhow, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio_tungstenite::connect_async;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info, warn};

/// Envelope from RTDS (§7.4.6).
#[derive(Debug, Deserialize)]
struct RtdsEnvelope {
    topic: String,
    #[serde(rename = "type")]
    #[allow(dead_code)]
    kind: String,
    timestamp: Option<u64>,
    payload: Value,
}

/// Dedup key (§7.4.9).
#[derive(Hash, Eq, PartialEq)]
struct DedupKey {
    topic: RtdsTopic,
    symbol: String,
    ts_payload_ms: u64,
    value_bits: u64,
}

pub struct RtdsClient {
    cfg: RtdsConfig,
    bus: EventSender,
    health: Arc<HealthMonitor>,
    /// Bounded dedup cache; we keep only recent keys via a ring-style VecDeque.
    recent: Arc<Mutex<DedupCache>>,
}

struct DedupCache {
    set: std::collections::HashSet<DedupKey>,
    order: std::collections::VecDeque<DedupKey>,
    cap: usize,
}

impl DedupCache {
    fn new(cap: usize) -> Self {
        Self {
            set: std::collections::HashSet::with_capacity(cap),
            order: std::collections::VecDeque::with_capacity(cap),
            cap,
        }
    }
    /// Returns true if `key` is new (not a duplicate).
    fn insert(&mut self, key: DedupKey) -> bool {
        if self.set.contains(&key) {
            return false;
        }
        let cloned = DedupKey {
            topic: key.topic,
            symbol: key.symbol.clone(),
            ts_payload_ms: key.ts_payload_ms,
            value_bits: key.value_bits,
        };
        self.order.push_back(key);
        self.set.insert(cloned);
        if self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
            }
        }
        true
    }
}

impl RtdsClient {
    pub fn new(cfg: RtdsConfig, bus: EventSender, health: Arc<HealthMonitor>) -> Self {
        Self {
            cfg,
            bus,
            health,
            recent: Arc::new(Mutex::new(DedupCache::new(4096))),
        }
    }

    /// Runs the connect/subscribe/read loop forever, reconnecting on drop.
    pub async fn run(self: Arc<Self>) -> Result<()> {
        let mut backoff = Backoff::new(&self.cfg.reconnect);
        loop {
            match self.run_once().await {
                Ok(()) => {
                    warn!(target: "rtds", "connection closed cleanly, reconnecting");
                }
                Err(e) => {
                    error!(target: "rtds", error = %e, "rtds loop error");
                    self.health.mark_rtds_degraded("loop_error");
                }
            }
            let d = backoff.next_delay();
            debug!(target: "rtds", delay_ms = d.as_millis() as u64, "reconnect backoff");
            tokio::time::sleep(d).await;
        }
    }

    async fn run_once(&self) -> Result<()> {
        info!(target: "rtds", url = %self.cfg.url, "connecting");
        let (ws, _) = connect_async(&self.cfg.url)
            .await
            .context("rtds connect_async")?;
        let (mut write, mut read) = ws.split();

        // Send the initial subscribe envelope (§7.4.3 / §7.4.4).
        let subscribe = build_subscribe_envelope(&self.cfg.subscriptions);
        write
            .send(Message::Text(subscribe.to_string()))
            .await
            .context("send subscribe")?;
        info!(target: "rtds", "subscribe sent: {}", subscribe);

        // Ping task — literal "PING" every 5s (§7.4.5).
        let ping_interval = Duration::from_millis(self.cfg.ping_interval_ms);
        let (ping_tx, mut ping_rx) = tokio::sync::mpsc::channel::<()>(1);
        let ping_task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(ping_interval);
            tick.tick().await; // consume immediate first tick
            loop {
                tick.tick().await;
                if ping_tx.send(()).await.is_err() {
                    break;
                }
            }
        });

        // Reset health because we successfully connected.
        self.health.mark_rtds_connected();

        loop {
            tokio::select! {
                maybe_msg = read.next() => {
                    match maybe_msg {
                        Some(Ok(Message::Text(t))) => {
                            if let Err(e) = self.handle_text(&t).await {
                                warn!(target: "rtds", error = %e, "parse error");
                                self.health.increment_rtds_parse_errors();
                            }
                        }
                        Some(Ok(Message::Binary(_))) => { /* ignore */ }
                        Some(Ok(Message::Ping(p))) => {
                            let _ = write.send(Message::Pong(p)).await;
                        }
                        Some(Ok(Message::Pong(_))) => {}
                        Some(Ok(Message::Frame(_))) => {}
                        Some(Ok(Message::Close(_))) | None => {
                            ping_task.abort();
                            return Ok(());
                        }
                        Some(Err(e)) => {
                            ping_task.abort();
                            return Err(anyhow!("ws read error: {e}"));
                        }
                    }
                }
                _ = ping_rx.recv() => {
                    if let Err(e) = write.send(Message::Text("PING".to_string())).await {
                        ping_task.abort();
                        return Err(anyhow!("ping send failed: {e}"));
                    }
                }
            }
        }
    }

    async fn handle_text(&self, text: &str) -> Result<()> {
        // Servers may send "PONG" or other non-JSON control strings; ignore.
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("PONG") || trimmed.is_empty() {
            return Ok(());
        }
        let env: RtdsEnvelope = serde_json::from_str(text)
            .with_context(|| format!("decode envelope: {trimmed:.200}"))?;

        let Some(topic) = map_topic(&env.topic) else {
            debug!(target: "rtds", topic = %env.topic, "ignoring unknown topic");
            return Ok(());
        };

        // Extract crypto payload fields (§7.4.7).
        let symbol = env
            .payload
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let ts_payload_ms = env
            .payload
            .get("timestamp")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let value = env
            .payload
            .get("value")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| anyhow!("missing payload.value"))?;

        let ts_server_ms = env.timestamp.unwrap_or_else(now_ms);
        let ts_recv_local_ms = now_ms();

        // Dedup (§7.4.9).
        let key = DedupKey {
            topic,
            symbol: symbol.clone(),
            ts_payload_ms,
            value_bits: value.to_bits(),
        };
        {
            let mut cache = self.recent.lock().await;
            if !cache.insert(key) {
                return Ok(());
            }
        }

        let update = RtdsCryptoUpdate {
            topic,
            symbol,
            ts_server_ms,
            ts_payload_ms,
            ts_recv_local_ms,
            value,
        };

        self.health
            .record_rtds_update(topic, ts_recv_local_ms);

        // Best-effort publish: if no subscribers, that's fine.
        let _ = self.bus.send(ExternalEvent::RtdsCrypto(update));
        Ok(())
    }
}

fn map_topic(raw: &str) -> Option<RtdsTopic> {
    match raw {
        "crypto_prices" => Some(RtdsTopic::Binance),
        "crypto_prices_chainlink" => Some(RtdsTopic::Chainlink),
        _ => None,
    }
}

fn build_subscribe_envelope(subs: &[RtdsSubscriptionConfig]) -> Value {
    let list: Vec<Value> = subs
        .iter()
        .map(|s| {
            let mut m = serde_json::Map::new();
            m.insert("topic".to_string(), json!(s.topic));
            m.insert("type".to_string(), json!(s.kind));
            if let Some(f) = &s.filters {
                m.insert("filters".to_string(), json!(f));
            }
            Value::Object(m)
        })
        .collect();
    json!({
        "action": "subscribe",
        "subscriptions": list,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_builder_matches_spec_shape() {
        let subs = vec![
            RtdsSubscriptionConfig {
                topic: "crypto_prices".into(),
                kind: "update".into(),
                filters: Some("btcusdt".into()),
            },
            RtdsSubscriptionConfig {
                topic: "crypto_prices_chainlink".into(),
                kind: "*".into(),
                filters: Some(r#"{"symbol":"btc/usd"}"#.into()),
            },
        ];
        let env = build_subscribe_envelope(&subs);
        assert_eq!(env["action"], "subscribe");
        let list = env["subscriptions"].as_array().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0]["topic"], "crypto_prices");
        assert_eq!(list[0]["type"], "update");
        assert_eq!(list[0]["filters"], "btcusdt");
        assert_eq!(list[1]["topic"], "crypto_prices_chainlink");
    }

    #[test]
    fn dedup_cache_rejects_duplicates() {
        let mut c = DedupCache::new(8);
        let k1 = DedupKey {
            topic: RtdsTopic::Binance,
            symbol: "btcusdt".into(),
            ts_payload_ms: 1,
            value_bits: 42,
        };
        let k1b = DedupKey {
            topic: RtdsTopic::Binance,
            symbol: "btcusdt".into(),
            ts_payload_ms: 1,
            value_bits: 42,
        };
        assert!(c.insert(k1));
        assert!(!c.insert(k1b));
    }

    #[test]
    fn map_topic_recognizes_spec_topics() {
        assert_eq!(map_topic("crypto_prices"), Some(RtdsTopic::Binance));
        assert_eq!(
            map_topic("crypto_prices_chainlink"),
            Some(RtdsTopic::Chainlink)
        );
        assert_eq!(map_topic("other"), None);
    }
}
