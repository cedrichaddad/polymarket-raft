//! Parametric fair-value backbone (§11.1).
//!
//!   p_0(t) = Phi( (log(S_t / K) - b(tau)) / sigma(tau) )
//!
//! Inputs:
//!   * S_t         : current Chainlink price
//!   * K           : opening reference price at window start
//!   * tau_seconds : seconds to resolution
//!   * sigma_per_sec : short-horizon scale; we scale by sqrt(tau) downstream
//!
//! `b(tau)` defaults to 0 (no drift). `sigma(tau)` is passed in explicitly —
//! the doc is clear that this is fitted from realized data, not Black–Scholes.

#[derive(Debug, Clone, Copy)]
pub struct FairValueParams {
    pub drift: f64,
    /// Per-second stdev of log returns. Caller scales by sqrt(tau) internally.
    pub sigma_per_sec: f64,
}

impl Default for FairValueParams {
    fn default() -> Self {
        Self {
            drift: 0.0,
            sigma_per_sec: 0.0005,
        }
    }
}

pub fn fair_prob_backbone(s_t: f64, k: f64, tau_seconds: f64, params: FairValueParams) -> Option<f64> {
    if s_t <= 0.0 || k <= 0.0 || tau_seconds <= 0.0 || params.sigma_per_sec <= 0.0 {
        return None;
    }
    let sigma_tau = params.sigma_per_sec * tau_seconds.sqrt();
    let b_tau = params.drift * tau_seconds;
    let z = ((s_t / k).ln() - b_tau) / sigma_tau;
    Some(standard_normal_cdf(z))
}

/// Abramowitz & Stegun 7.1.26 approximation. Accurate to ~7.5e-8.
pub fn standard_normal_cdf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x_abs = x.abs() / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + p * x_abs);
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x_abs * x_abs).exp();
    0.5 * (1.0 + sign * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn at_the_money_is_half() {
        let p = fair_prob_backbone(100.0, 100.0, 60.0, FairValueParams::default()).unwrap();
        assert!((p - 0.5).abs() < 1e-9);
    }

    #[test]
    fn above_reference_is_above_half() {
        let p = fair_prob_backbone(101.0, 100.0, 60.0, FairValueParams::default()).unwrap();
        assert!(p > 0.5);
    }

    #[test]
    fn returns_none_for_invalid_inputs() {
        assert!(fair_prob_backbone(0.0, 100.0, 60.0, FairValueParams::default()).is_none());
        assert!(fair_prob_backbone(100.0, 100.0, 0.0, FairValueParams::default()).is_none());
    }

    #[test]
    fn phi_matches_known_values() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((standard_normal_cdf(1.0) - 0.8413447).abs() < 1e-4);
        assert!((standard_normal_cdf(-1.0) - 0.1586553).abs() < 1e-4);
    }
}
