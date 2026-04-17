//! Calibration layer (§11.2).
//!
//! v1 ships with an `IsotonicCalibrator` fitted from (p_0, y) pairs via the
//! pool-adjacent-violators algorithm. More complex calibrators (logistic,
//! GBT) are added later; they just implement the `Calibrator` trait.
//!
//! The isotonic model is persisted as the vector of breakpoints — load
//! from JSON at service start using the Python research pipeline as
//! ground truth.

use serde::{Deserialize, Serialize};

pub trait Calibrator: Send + Sync {
    /// Map a raw backbone probability in [0,1] to a calibrated one in [0,1].
    fn apply(&self, p_raw: f64) -> f64;
}

/// Isotonic calibrator: stores a sorted list of (x, y) breakpoints with
/// monotone-nondecreasing y. Linear interpolation between breakpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicCalibrator {
    pub breakpoints: Vec<(f64, f64)>,
}

impl IsotonicCalibrator {
    pub fn identity() -> Self {
        Self {
            breakpoints: vec![(0.0, 0.0), (1.0, 1.0)],
        }
    }

    /// Fit via the pool-adjacent-violators algorithm on `(x, y)` with optional
    /// sample weights. Inputs are assumed in any order; we sort by x.
    pub fn fit(xs: &[f64], ys: &[f64], weights: Option<&[f64]>) -> Self {
        assert_eq!(xs.len(), ys.len());
        if xs.is_empty() {
            return Self::identity();
        }
        let w = weights.map(|w| w.to_vec()).unwrap_or_else(|| vec![1.0; xs.len()]);

        // Sort by x.
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        idx.sort_by(|&a, &b| xs[a].partial_cmp(&xs[b]).unwrap_or(std::cmp::Ordering::Equal));

        // PAV: maintain running pools.
        let mut pool_x: Vec<f64> = Vec::new();
        let mut pool_y: Vec<f64> = Vec::new();
        let mut pool_w: Vec<f64> = Vec::new();
        for &i in &idx {
            pool_x.push(xs[i]);
            pool_y.push(ys[i]);
            pool_w.push(w[i]);
            while pool_y.len() >= 2 {
                let n = pool_y.len();
                if pool_y[n - 2] > pool_y[n - 1] {
                    let y1 = pool_y.pop().unwrap();
                    let w1 = pool_w.pop().unwrap();
                    let x1 = pool_x.pop().unwrap();
                    let y0 = pool_y.pop().unwrap();
                    let w0 = pool_w.pop().unwrap();
                    let x0 = pool_x.pop().unwrap();
                    let w_sum = w0 + w1;
                    pool_y.push((y0 * w0 + y1 * w1) / w_sum);
                    pool_w.push(w_sum);
                    pool_x.push(x0.min(x1)); // representative x; collapse to leftmost
                    let _ = x1;
                } else {
                    break;
                }
            }
        }

        let mut bp: Vec<(f64, f64)> = pool_x.into_iter().zip(pool_y).collect();
        bp.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Ensure endpoints exist so interpolation is well-defined in [0,1].
        if bp.first().map(|(x, _)| *x > 0.0).unwrap_or(true) {
            let y0 = bp.first().map(|p| p.1).unwrap_or(0.0);
            bp.insert(0, (0.0, y0));
        }
        if bp.last().map(|(x, _)| *x < 1.0).unwrap_or(true) {
            let y1 = bp.last().map(|p| p.1).unwrap_or(1.0);
            bp.push((1.0, y1));
        }
        Self { breakpoints: bp }
    }
}

impl Calibrator for IsotonicCalibrator {
    fn apply(&self, p_raw: f64) -> f64 {
        let x = p_raw.clamp(0.0, 1.0);
        let bp = &self.breakpoints;
        if bp.is_empty() {
            return x;
        }
        // Binary-search bracket.
        match bp.binary_search_by(|probe| probe.0.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => bp[i].1,
            Err(i) => {
                if i == 0 {
                    bp[0].1
                } else if i >= bp.len() {
                    bp[bp.len() - 1].1
                } else {
                    let (x0, y0) = bp[i - 1];
                    let (x1, y1) = bp[i];
                    if (x1 - x0).abs() < 1e-12 {
                        y0
                    } else {
                        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_passes_through() {
        let c = IsotonicCalibrator::identity();
        assert!((c.apply(0.3) - 0.3).abs() < 1e-9);
        assert!((c.apply(0.95) - 0.95).abs() < 1e-9);
    }

    #[test]
    fn fit_respects_monotonicity_on_noisy_input() {
        // True map y = x but with violations.
        let xs = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let ys = vec![0.0, 0.3, 0.2, 0.5, 0.4, 0.6, 0.8, 0.7, 1.0];
        let c = IsotonicCalibrator::fit(&xs, &ys, None);
        let bp = &c.breakpoints;
        for w in bp.windows(2) {
            assert!(w[1].1 >= w[0].1 - 1e-9);
        }
        // Endpoints clamp roughly to input range.
        assert!(c.apply(0.1) >= 0.0);
        assert!(c.apply(0.9) <= 1.0);
    }
}
