//! Signed-flow aggregates over a trailing window.

use crate::features::live_state::TradeSample;

pub fn signed_flow_secs(trades: &[TradeSample], now_ms: u64, window_secs: u64) -> f64 {
    let cutoff = now_ms.saturating_sub(window_secs.saturating_mul(1000));
    trades
        .iter()
        .filter(|t| t.ts_ms >= cutoff)
        .map(|t| t.signed_size)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flow_sums_within_window() {
        let ts = vec![
            TradeSample { ts_ms: 1_000, signed_size: 1.0, price: 0.5 },
            TradeSample { ts_ms: 2_000, signed_size: -0.5, price: 0.49 },
            TradeSample { ts_ms: 3_000, signed_size: 2.0, price: 0.51 },
        ];
        // Window ending at t=3.5s, 2s back => includes t>=1.5s.
        let f = signed_flow_secs(&ts, 3_500, 2);
        assert!((f - 1.5).abs() < 1e-9);
    }
}
