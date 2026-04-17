//! Liveness heartbeat (§13.3, §19.2).
//!
//! If the heartbeat isn't ticked within `deadline_ms`, the trip callback
//! fires — typically `quote_manager.force_cancel_all()` and
//! `risk_manager.kill("heartbeat_timeout")`.

use crate::time_util::now_ms;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Heartbeat {
    last_tick_ms: AtomicU64,
    deadline_ms: u64,
    pub tripped: RwLock<Option<String>>,
}

impl Heartbeat {
    pub fn new(deadline_ms: u64) -> Self {
        Self {
            last_tick_ms: AtomicU64::new(now_ms()),
            deadline_ms,
            tripped: RwLock::new(None),
        }
    }

    pub fn tick(&self) {
        self.last_tick_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Returns `Some(reason)` if the heartbeat has tripped.
    pub fn check(&self) -> Option<String> {
        if let Some(r) = self.tripped.read().clone() {
            return Some(r);
        }
        let since = now_ms().saturating_sub(self.last_tick_ms.load(Ordering::Relaxed));
        if since > self.deadline_ms {
            let reason = format!("heartbeat_timeout_{since}ms");
            *self.tripped.write() = Some(reason.clone());
            return Some(reason);
        }
        None
    }
}
