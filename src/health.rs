//! Feed health monitor (§7.7.4, §7.4.11).
//!
//! Tracks per-feed freshness and parse-error counts so the execution layer
//! can refuse to trade on stale or degraded feeds.

use crate::time_util::now_ms;
use crate::types::RtdsTopic;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Default)]
pub struct FeedHealthSnapshot {
    pub rtds_connected: bool,
    pub binance_last_update_ms: Option<u64>,
    pub chainlink_last_update_ms: Option<u64>,
    pub binance_stale_ms: u64,
    pub chainlink_stale_ms: u64,
    pub parse_errors: u64,
    pub degraded_reason: Option<String>,
}

#[derive(Debug, Default)]
pub struct HealthMonitor {
    rtds_connected: RwLock<bool>,
    binance_last: AtomicU64,   // 0 => None
    chainlink_last: AtomicU64, // 0 => None
    parse_errors: AtomicU64,
    degraded_reason: RwLock<Option<String>>,
    // Stale thresholds
    binance_stale_ms: AtomicU64,
    chainlink_stale_ms: AtomicU64,
}

impl HealthMonitor {
    pub fn new(binance_stale_ms: u64, chainlink_stale_ms: u64) -> Self {
        Self {
            rtds_connected: RwLock::new(false),
            binance_last: AtomicU64::new(0),
            chainlink_last: AtomicU64::new(0),
            parse_errors: AtomicU64::new(0),
            degraded_reason: RwLock::new(None),
            binance_stale_ms: AtomicU64::new(binance_stale_ms),
            chainlink_stale_ms: AtomicU64::new(chainlink_stale_ms),
        }
    }

    pub fn mark_rtds_connected(&self) {
        *self.rtds_connected.write() = true;
        *self.degraded_reason.write() = None;
    }

    pub fn mark_rtds_degraded(&self, reason: &str) {
        *self.rtds_connected.write() = false;
        *self.degraded_reason.write() = Some(reason.to_string());
    }

    pub fn increment_rtds_parse_errors(&self) {
        self.parse_errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_rtds_update(&self, topic: RtdsTopic, recv_ms: u64) {
        match topic {
            RtdsTopic::Binance => self.binance_last.store(recv_ms, Ordering::Relaxed),
            RtdsTopic::Chainlink => self.chainlink_last.store(recv_ms, Ordering::Relaxed),
        }
    }

    pub fn snapshot(&self) -> FeedHealthSnapshot {
        let _now = now_ms();
        let b = self.binance_last.load(Ordering::Relaxed);
        let c = self.chainlink_last.load(Ordering::Relaxed);
        FeedHealthSnapshot {
            rtds_connected: *self.rtds_connected.read(),
            binance_last_update_ms: (b != 0).then_some(b),
            chainlink_last_update_ms: (c != 0).then_some(c),
            binance_stale_ms: self.binance_stale_ms.load(Ordering::Relaxed),
            chainlink_stale_ms: self.chainlink_stale_ms.load(Ordering::Relaxed),
            parse_errors: self.parse_errors.load(Ordering::Relaxed),
            degraded_reason: self.degraded_reason.read().clone(),
        }
    }

    pub fn binance_stale(&self) -> bool {
        let b = self.binance_last.load(Ordering::Relaxed);
        if b == 0 {
            return true;
        }
        now_ms().saturating_sub(b) > self.binance_stale_ms.load(Ordering::Relaxed)
    }

    pub fn chainlink_stale(&self) -> bool {
        let c = self.chainlink_last.load(Ordering::Relaxed);
        if c == 0 {
            return true;
        }
        now_ms().saturating_sub(c) > self.chainlink_stale_ms.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_until_we_get_an_update() {
        let h = HealthMonitor::new(2000, 5000);
        assert!(h.binance_stale());
        assert!(h.chainlink_stale());
        h.record_rtds_update(RtdsTopic::Binance, now_ms());
        assert!(!h.binance_stale());
        // chainlink still stale until recorded
        assert!(h.chainlink_stale());
    }
}
