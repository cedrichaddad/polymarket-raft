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
        // We query the bitcoin tag to get a more targeted result set.
        let url = format!(
            "{}/events?tag_slug=bitcoin&active=true&closed=false&limit=100",
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
                // Log what we see before filtering so we can tune infer_window.
                let slug = m.get("slug").and_then(|x| x.as_str()).unwrap_or("-");
                let title = m.get("question").and_then(|x| x.as_str()).unwrap_or("-");
                debug!(target: "hydrator", slug, title = &title[..title.len().min(80)], "candidate market");
                let Some(meta) = parse_market(&m) else { continue };
                if !is_btc_minute(&meta) {
                    continue;
                }
                let dur_s = meta.end_ts_ms.saturating_sub(meta.start_ts_ms) / 1000;
                debug!(target: "hydrator", market = %meta.market_id, window = %meta.window_type, dur_s, "matched btc minute market");
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

    let slug = m.get("slug").and_then(|x| x.as_str()).unwrap_or_default();
    let title = m.get("question").and_then(|x| x.as_str()).unwrap_or_default();

    // The canonical BTC up/down minute market slug encodes the window start time
    // and duration: btc-updown-{5m|15m}-{unix_seconds}. The Gamma API's
    // startDate/endDate span the full batch lifecycle (~24h), NOT the 5/15-minute
    // prediction window. We derive the correct timestamps from the slug.
    let (window_type, start_ts_ms, end_ts_ms) =
        if let Some((wt, s, e)) = parse_btc_updown_slug(slug) {
            (wt, s, e)
        } else {
            let wt = infer_window(slug, title).unwrap_or_else(|| "unknown".to_string());
            let s = parse_ts(m.get("startDate")).unwrap_or(0);
            let e = parse_ts(m.get("endDate")).unwrap_or(0);
            (wt, s, e)
        };

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

/// Parse the canonical BTC up/down slug format: `btc-updown-{5m|15m}-{unix_seconds}`.
/// Returns `(window_type, start_ts_ms, end_ts_ms)` on match, None otherwise.
///
/// The Gamma API's startDate/endDate cover the full ~24-hour market lifecycle;
/// the actual prediction window is encoded only in the slug timestamp.
fn parse_btc_updown_slug(slug: &str) -> Option<(String, u64, u64)> {
    // Split into exactly 4 parts on the first 3 hyphens.
    let mut it = slug.splitn(4, '-');
    if it.next() != Some("btc") { return None; }
    if it.next() != Some("updown") { return None; }
    let window = it.next()?;
    let ts_str = it.next()?;
    let (window_type, duration_ms): (&str, u64) = match window {
        "5m"  => ("btc_5m",  5  * 60 * 1_000),
        "15m" => ("btc_15m", 15 * 60 * 1_000),
        _     => return None,  // 4h and other windows are not our target
    };
    let unix_s: u64 = ts_str.parse().ok()?;
    Some((window_type.to_string(), unix_s * 1_000, unix_s * 1_000 + duration_ms))
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

    #[test]
    fn btc_updown_slug_parses_correctly() {
        let ts = 1776476700u64;
        let (wt, s, e) = parse_btc_updown_slug("btc-updown-5m-1776476700").unwrap();
        assert_eq!(wt, "btc_5m");
        assert_eq!(s, ts * 1_000);
        assert_eq!(e, ts * 1_000 + 300_000);

        let (wt, s, e) = parse_btc_updown_slug("btc-updown-15m-1776476700").unwrap();
        assert_eq!(wt, "btc_15m");
        assert_eq!(e - s, 900_000);

        // 4h windows should not match.
        assert!(parse_btc_updown_slug("btc-updown-4h-1776470400").is_none());
        // Non-btc-updown slugs should not match.
        assert!(parse_btc_updown_slug("bitcoin-above-70k-on-april-18").is_none());
    }
}
