//! All-in taker cost threshold (§12.1).
//!
//!   Delta_all_in(t) = spread + fee + slippage + adverse + model
//!
//! All terms are expressed in *probability* units (0..1), not cents or USDC.
//! A crossing cost of 1 tick on a contract that trades in 0.01 ticks contributes
//! 0.005 to `spread` (half-spread). Keeping a single unit system lets the
//! router compare `|p_hybrid - p_mkt|` directly to a single scalar threshold.

#[derive(Debug, Clone, Copy)]
pub struct AllInCost {
    pub spread: f64,
    pub fee: f64,
    pub slippage: f64,
    pub adverse: f64,
    pub model: f64,
}

impl AllInCost {
    pub fn total(&self) -> f64 {
        self.spread + self.fee + self.slippage + self.adverse + self.model
    }
}

impl Default for AllInCost {
    fn default() -> Self {
        Self {
            spread: 0.005,
            fee: 0.0,
            slippage: 0.001,
            adverse: 0.003,
            model: 0.005,
        }
    }
}

/// Expected-value gate for maker quoting (§12.2):
///     E[spread capture] - E[adverse] + E[rebate] > 0
pub fn maker_expected_value(
    expected_spread_capture: f64,
    expected_adverse: f64,
    expected_rebate: f64,
) -> f64 {
    expected_spread_capture - expected_adverse + expected_rebate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn total_sums_terms() {
        let c = AllInCost { spread: 0.001, fee: 0.002, slippage: 0.0, adverse: 0.0, model: 0.0 };
        assert!((c.total() - 0.003).abs() < 1e-12);
    }
}
