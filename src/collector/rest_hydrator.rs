//! REST hydrator (§7.6, §7.7.3).
//!
//! Polls the Polymarket Gamma API for active BTC 5m / 15m markets and hydrates
//! `MarketMetaEvent`s onto the bus. v1 runs on startup and every
//! `hydration.refresh_secs`.
//!
//! Because the exact Gamma schema for minute markets evolves, this module
//! keeps a tolerant parser: required fields surface errors, optional fields
//! degrade gracefully.

use crate::config::HydrationConfig;
use crate::event_bus::EventSender;
use crate::types::{ExternalEvent, MarketMetaEvent};

use anyhow::{Context, Result};
use chrono::DateTime;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

pub struct RestHydrator {
    cfg: HydrationConfig,
    bus: EventSender,
    http: Client,
    /// Cache of market_ids we've already seen so we can detect changes.
    known: Arc<RwLock<HashSet<String>>>,
    /// Latest asset_id set (both yes/no tokens) aggregated across discovered markets.
    latest_assets: Arc<RwLock<Vec<String>>>,
}

impl RestHydrator {
    pub fn new(cfg: HydrationConfig, bus: EventSender) -> Self {
        Self {
            cfg,
            bus,
            http: Client::builder()
                .user_agent("raft/0.1")
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("reqwest client"),
            known: Arc::new(RwLock::new(HashSet::new())),
            latest_assets: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn current_assets(&self) -> Vec<String> {
        self.latest_assets.read().await.clone()
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let interval = std::time::Duration::from_secs(self.cfg.refresh_secs.max(5));
        loop {
            match self.refresh_once().await {
                Ok(n) => debug!(target: "hydrator", markets = n, "refreshed"),
                Err(e) => warn!(target: "hydrator", error = %e, "refresh failed"),
            }
            tokio::time::sleep(interval).await;
        }
    }

    pub async fn refresh_once(&self) -> Result<usize> {
        // Gamma `events` endpoint filtered for BTC minute markets.
        // We fetch both 5m and 15m series; filters may need adjustment as Gamma changes.
        let url = format!(
            "{}/events?tag_slug=crypto&active=true&closed=false&limit=100",
            self.cfg.gamma_url.trim_end_matches('/')
        );
        let resp = self.http.get(&url).send().await.context("gamma get")?;
        let text = resp.text().await.context("gamma body")?;
        let v: Value = serde_json::from_str(&text).context("gamma json")?;

        let events = v.as_array().cloned().unwrap_or_default();
        let mut count = 0usize;
        let mut asset_set: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for ev in events {
            let markets = ev
                .get("markets")
                .and_then(|x| x.as_array())
                .cloned()
                .unwrap_or_default();
            for m in markets {
                let Some(meta) = parse_market(&m) else { continue };
                if !is_btc_minute(&meta) {
                    continue;
                }
                asset_set.push(meta.asset_yes_id.clone());
                asset_set.push(meta.asset_no_id.clone());
                seen.insert(meta.market_id.clone());
                // Publish on first observation and on re-observation; downstream dedups.
                let _ = self.bus.send(ExternalEvent::MarketMeta(meta));
                count += 1;
            }
        }

        {
            let mut k = self.known.write().await;
            for id in seen.iter() {
                k.insert(id.clone());
            }
        }
        *self.latest_assets.write().await = asset_set;
        if count > 0 {
            info!(target: "hydrator", btc_minute_markets = count, "hydrated");
        }
        Ok(count)
    }
}

fn parse_market(m: &Value) -> Option<MarketMetaEvent> {
    let market_id = m.get("conditionId").or_else(|| m.get("id")).and_then(|x| x.as_str())?.to_string();
    // Polymarket exposes two token IDs per binary market (yes/no outcomes).
    let tokens = m
        .get("clobTokenIds")
        .and_then(|x| {
            // Some responses return a JSON-encoded string, others an array.
            if let Some(s) = x.as_str() {
                serde_json::from_str::<Value>(s).ok()
            } else {
                Some(x.clone())
            }
        })
        .and_then(|x| x.as_array().cloned())
        .unwrap_or_default();
    if tokens.len() < 2 {
        return None;
    }
    let asset_yes_id = tokens[0].as_str()?.to_string();
    let asset_no_id = tokens[1].as_str()?.to_string();

    // Window type inferred from slug/title if explicit fields are absent.
    let slug = m.get("slug").and_then(|x| x.as_str()).unwrap_or_default();
    let title = m.get("question").and_then(|x| x.as_str()).unwrap_or_default();
    let window_type = infer_window(slug, title).unwrap_or_else(|| "unknown".to_string());

    let start_ts_ms = parse_ts(m.get("startDate")).unwrap_or(0);
    let end_ts_ms = parse_ts(m.get("endDate")).unwrap_or(0);

    let fees_enabled = m
        .get("enableOrderBook")
        .and_then(|x| x.as_bool())
        .unwrap_or(true);
    let fee_schedule_json = m
        .get("fees")
        .map(|x| x.to_string())
        .unwrap_or_else(|| "{}".to_string());
    let tick_size = m.get("tickSize").and_then(|x| match x {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    });

    Some(MarketMetaEvent {
        market_id,
        asset_yes_id,
        asset_no_id,
        window_type,
        start_ts_ms,
        end_ts_ms,
        resolution_source: "chainlink_btc_usd".to_string(),
        fees_enabled,
        fee_schedule_json,
        tick_size,
    })
}

fn parse_ts(v: Option<&Value>) -> Option<u64> {
    let s = v?.as_str()?;
    DateTime::parse_from_rfc3339(s).ok().map(|t| t.timestamp_millis() as u64)
}

fn infer_window(slug: &str, title: &str) -> Option<String> {
    let lower = format!("{slug} {title}").to_lowercase();
    if !lower.contains("btc") && !lower.contains("bitcoin") {
        return None;
    }
    // Check 15-minute first so it doesn't get swallowed by the "5-minute" substring.
    if lower.contains("15-minute") || lower.contains("15 min") || lower.contains("15m") {
        Some("btc_15m".into())
    } else if lower.contains("5-minute") || lower.contains("5 min") || lower.contains("5m") {
        Some("btc_5m".into())
    } else {
        None
    }
}

fn is_btc_minute(m: &MarketMetaEvent) -> bool {
    matches!(m.window_type.as_str(), "btc_5m" | "btc_15m")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_inference_picks_up_common_patterns() {
        assert_eq!(
            infer_window("btc-up-or-down-5m-march-1-4pm", ""),
            Some("btc_5m".into())
        );
        assert_eq!(
            infer_window("", "Bitcoin up or down in 15 minutes?"),
            Some("btc_15m".into())
        );
        assert_eq!(infer_window("eth-5m", "Eth price move"), None);
    }
}
