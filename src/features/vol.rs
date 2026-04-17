//! Short-horizon realized vol estimators.
//!
//! `realized_vol_secs` returns the sample stdev of log returns computed from
//! the supplied reference-price buffer over the trailing `window_secs`.
//! Returns are first differences of `log(price)` — the result scales like per-
//! sample volatility, not annualized. We annualize only inside feature
//! construction if needed; keeping the primitive unit-free is easier to test.

use crate::features::live_state::RefPriceSample;

pub fn realized_vol_secs(samples: &[RefPriceSample], now_ms: u64, window_secs: u64) -> Option<f64> {
    if samples.len() < 3 {
        return None;
    }
    let cutoff = now_ms.saturating_sub(window_secs.saturating_mul(1000));
    let filtered: Vec<f64> = samples
        .iter()
        .filter(|s| s.ts_ms >= cutoff && s.value > 0.0)
        .map(|s| s.value.ln())
        .collect();
    if filtered.len() < 3 {
        return None;
    }
    let mut diffs = Vec::with_capacity(filtered.len() - 1);
    for w in filtered.windows(2) {
        diffs.push(w[1] - w[0]);
    }
    if diffs.len() < 2 {
        return None;
    }
    let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
    let var = diffs.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64;
    Some(var.sqrt())
}

pub fn recent_log_return(
    samples: &[RefPriceSample],
    now_ms: u64,
    window_secs: u64,
) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let cutoff = now_ms.saturating_sub(window_secs.saturating_mul(1000));
    let latest = samples.last()?;
    let first = samples
        .iter()
        .rev()
        .find(|s| s.ts_ms <= cutoff)
        .or_else(|| samples.first())?;
    if first.value <= 0.0 || latest.value <= 0.0 {
        return None;
    }
    Some((latest.value / first.value).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vol_is_zero_for_constant_prices() {
        let mut s = Vec::new();
        for i in 0..10 {
            s.push(RefPriceSample { ts_ms: i * 1000, value: 100.0 });
        }
        let v = realized_vol_secs(&s, 10_000, 30).unwrap();
        assert!(v.abs() < 1e-12);
    }

    #[test]
    fn vol_is_positive_when_prices_move() {
        let s: Vec<_> = (0..10)
            .map(|i| RefPriceSample {
                ts_ms: i * 1000,
                value: 100.0 * (1.0 + 0.01 * (i as f64 % 2.0)),
            })
            .collect();
        let v = realized_vol_secs(&s, 10_000, 30).unwrap();
        assert!(v > 0.0);
    }
}
