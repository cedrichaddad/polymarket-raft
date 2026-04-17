//! Exponential backoff with jitter, shared by all reconnecting collectors.
//! Matches the schedule in §7.4.10: 250ms, 500ms, 1s, 2s, cap 5s with jitter.

use crate::config::BackoffConfig;
use rand::Rng;
use std::time::Duration;

#[derive(Debug)]
pub struct Backoff {
    initial_ms: u64,
    max_ms: u64,
    jitter: bool,
    current_ms: u64,
}

impl Backoff {
    pub fn new(cfg: &BackoffConfig) -> Self {
        Self {
            initial_ms: cfg.initial_backoff_ms,
            max_ms: cfg.max_backoff_ms,
            jitter: cfg.jitter,
            current_ms: cfg.initial_backoff_ms,
        }
    }

    pub fn reset(&mut self) {
        self.current_ms = self.initial_ms;
    }

    pub fn next_delay(&mut self) -> Duration {
        let base = self.current_ms.min(self.max_ms);
        let delay = if self.jitter {
            // Full jitter: uniform in [base/2, base*3/2] clamped to [0, max].
            let low = base / 2;
            let high = base.saturating_add(base / 2);
            let mut rng = rand::thread_rng();
            rng.gen_range(low..=high).min(self.max_ms)
        } else {
            base
        };
        // Double for next call.
        self.current_ms = (self.current_ms.saturating_mul(2)).min(self.max_ms);
        Duration::from_millis(delay)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_monotonically_doubles_up_to_cap() {
        let cfg = BackoffConfig {
            initial_backoff_ms: 100,
            max_backoff_ms: 1000,
            jitter: false,
        };
        let mut b = Backoff::new(&cfg);
        assert_eq!(b.next_delay().as_millis(), 100);
        assert_eq!(b.next_delay().as_millis(), 200);
        assert_eq!(b.next_delay().as_millis(), 400);
        assert_eq!(b.next_delay().as_millis(), 800);
        assert_eq!(b.next_delay().as_millis(), 1000);
        assert_eq!(b.next_delay().as_millis(), 1000);
        b.reset();
        assert_eq!(b.next_delay().as_millis(), 100);
    }

    #[test]
    fn jitter_stays_within_bounds() {
        let cfg = BackoffConfig {
            initial_backoff_ms: 200,
            max_backoff_ms: 5000,
            jitter: true,
        };
        let mut b = Backoff::new(&cfg);
        for _ in 0..50 {
            let d = b.next_delay().as_millis() as u64;
            assert!(d <= 5000);
        }
    }
}
