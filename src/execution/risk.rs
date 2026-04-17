//! Risk manager (§19).
//!
//! Enforces hard and soft risk limits before any `OrderIntent` leaves the
//! service. Counters are kept in memory for v1 — a daily loss reset would be
//! triggered externally (cron / orchestration).

use parking_lot::RwLock;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct RiskConfig {
    pub max_notional_per_market: f64,
    pub max_directional_per_family: f64,
    pub max_open_orders: usize,
    pub max_daily_loss_usdc: f64,
    pub reject_streak_limit: u32,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_notional_per_market: 100.0,
            max_directional_per_family: 500.0,
            max_open_orders: 16,
            max_daily_loss_usdc: 50.0,
            reject_streak_limit: 5,
        }
    }
}

#[derive(Debug, Default)]
struct RiskStateInner {
    notional_by_market: HashMap<String, f64>,
    directional_by_family: HashMap<String, f64>,
    open_orders: usize,
    day_loss_usdc: f64,
    reject_streak: u32,
    killed: Option<String>,
}

pub struct RiskManager {
    cfg: RiskConfig,
    state: RwLock<RiskStateInner>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RiskVerdict {
    Ok,
    Reject(String),
}

impl RiskManager {
    pub fn new(cfg: RiskConfig) -> Self {
        Self {
            cfg,
            state: RwLock::new(RiskStateInner::default()),
        }
    }

    pub fn kill(&self, reason: impl Into<String>) {
        self.state.write().killed = Some(reason.into());
    }

    pub fn is_killed(&self) -> Option<String> {
        self.state.read().killed.clone()
    }

    pub fn check_new_order(
        &self,
        market_id: &str,
        family: &str,
        notional: f64,
        directional_signed: f64,
    ) -> RiskVerdict {
        let s = self.state.read();
        if let Some(reason) = &s.killed {
            return RiskVerdict::Reject(format!("killed: {reason}"));
        }
        if s.open_orders >= self.cfg.max_open_orders {
            return RiskVerdict::Reject("max_open_orders".into());
        }
        if s.day_loss_usdc >= self.cfg.max_daily_loss_usdc {
            return RiskVerdict::Reject("daily_loss_cap".into());
        }
        let m_notional = s.notional_by_market.get(market_id).copied().unwrap_or(0.0);
        if m_notional + notional > self.cfg.max_notional_per_market {
            return RiskVerdict::Reject("market_notional_cap".into());
        }
        let fam = s.directional_by_family.get(family).copied().unwrap_or(0.0);
        if (fam + directional_signed).abs() > self.cfg.max_directional_per_family {
            return RiskVerdict::Reject("family_directional_cap".into());
        }
        RiskVerdict::Ok
    }

    pub fn on_order_accepted(&self, market_id: &str, family: &str, notional: f64, directional_signed: f64) {
        let mut s = self.state.write();
        s.open_orders += 1;
        *s.notional_by_market.entry(market_id.to_string()).or_insert(0.0) += notional;
        *s.directional_by_family.entry(family.to_string()).or_insert(0.0) += directional_signed;
        s.reject_streak = 0;
    }

    pub fn on_order_rejected(&self) {
        let mut s = self.state.write();
        s.reject_streak += 1;
        if s.reject_streak >= self.cfg.reject_streak_limit {
            s.killed = Some(format!("reject_streak={}", s.reject_streak));
        }
    }

    pub fn on_order_closed(&self, market_id: &str) {
        let mut s = self.state.write();
        if s.open_orders > 0 {
            s.open_orders -= 1;
        }
        if let Some(v) = s.notional_by_market.get_mut(market_id) {
            *v = (*v - 1.0).max(0.0);
        }
    }

    pub fn on_realized_pnl(&self, pnl_usdc: f64) {
        let mut s = self.state.write();
        if pnl_usdc < 0.0 {
            s.day_loss_usdc += -pnl_usdc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_switch_rejects_new_orders() {
        let r = RiskManager::new(RiskConfig::default());
        r.kill("test");
        let v = r.check_new_order("m", "btc_5m", 10.0, 1.0);
        assert!(matches!(v, RiskVerdict::Reject(_)));
    }

    #[test]
    fn market_notional_cap_blocks_oversized_orders() {
        let cfg = RiskConfig { max_notional_per_market: 50.0, ..Default::default() };
        let r = RiskManager::new(cfg);
        r.on_order_accepted("m", "btc_5m", 40.0, 0.0);
        let v = r.check_new_order("m", "btc_5m", 20.0, 0.0);
        assert!(matches!(v, RiskVerdict::Reject(_)));
    }
}
