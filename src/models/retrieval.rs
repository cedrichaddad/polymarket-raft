//! Nearest-neighbor retrieval interface (§17).
//!
//! The canonical retrieval store is Vibrato (§8). The Rust service queries it
//! through this trait. For v1 we ship a tiny reference implementation
//! (`BruteForceRetrieval`) that scans a slice of in-memory vectors — useful in
//! tests and when running the live loop against a small backtest set without
//! spinning up Vibrato. Swap in the real implementation by providing another
//! type that satisfies `Retrieval`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalQuery {
    /// State vector (§10.1).
    pub vector: Vec<f64>,
    /// Top-k to retrieve.
    pub k: usize,
    /// Optional market family filter, e.g. "btc_5m".
    pub market_family: Option<String>,
    /// Optional tick-regime tag: "center" / "wing".
    pub tick_regime: Option<String>,
    /// Optional volatility bucket tag: "low" / "med" / "high".
    pub vol_regime: Option<String>,
    /// Only include neighbors whose `sequence_ts` is within this window
    /// (inclusive, milliseconds). None => unrestricted.
    pub min_sequence_ts_ms: Option<u64>,
    pub max_sequence_ts_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalHit {
    pub sequence_ts_ms: u64,
    pub distance: f64,
    /// Realized UP label for this state (0 / 1).
    pub resolved_up: Option<f64>,
    /// Pointer into the canonical store so caller can hydrate richer context.
    pub row_id: Option<String>,
}

pub trait Retrieval: Send + Sync {
    fn query(&self, q: &RetrievalQuery) -> Vec<RetrievalHit>;
}

/// Item stored in the brute-force reference index.
#[derive(Debug, Clone)]
pub struct RetrievalItem {
    pub sequence_ts_ms: u64,
    pub vector: Vec<f64>,
    pub market_family: Option<String>,
    pub tick_regime: Option<String>,
    pub vol_regime: Option<String>,
    pub resolved_up: Option<f64>,
    pub row_id: Option<String>,
}

pub struct BruteForceRetrieval {
    items: Vec<RetrievalItem>,
}

impl BruteForceRetrieval {
    pub fn new(items: Vec<RetrievalItem>) -> Self {
        Self { items }
    }

    fn passes_filter(item: &RetrievalItem, q: &RetrievalQuery) -> bool {
        if let Some(f) = &q.market_family {
            if item.market_family.as_deref() != Some(f) {
                return false;
            }
        }
        if let Some(f) = &q.tick_regime {
            if item.tick_regime.as_deref() != Some(f) {
                return false;
            }
        }
        if let Some(f) = &q.vol_regime {
            if item.vol_regime.as_deref() != Some(f) {
                return false;
            }
        }
        if let Some(lo) = q.min_sequence_ts_ms {
            if item.sequence_ts_ms < lo {
                return false;
            }
        }
        if let Some(hi) = q.max_sequence_ts_ms {
            if item.sequence_ts_ms > hi {
                return false;
            }
        }
        true
    }
}

impl Retrieval for BruteForceRetrieval {
    fn query(&self, q: &RetrievalQuery) -> Vec<RetrievalHit> {
        let mut scored: Vec<(f64, &RetrievalItem)> = self
            .items
            .iter()
            .filter(|it| Self::passes_filter(it, q))
            .filter(|it| it.vector.len() == q.vector.len())
            .map(|it| (l2_sq(&it.vector, &q.vector), it))
            .collect();
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored
            .into_iter()
            .take(q.k)
            .map(|(d, it)| RetrievalHit {
                sequence_ts_ms: it.sequence_ts_ms,
                distance: d.sqrt(),
                resolved_up: it.resolved_up,
                row_id: it.row_id.clone(),
            })
            .collect()
    }
}

fn l2_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Helper: p_nn per §11.3 — mean realized UP over retrieved neighbors.
pub fn p_nn(hits: &[RetrievalHit]) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for h in hits {
        if let Some(y) = h.resolved_up {
            sum += y;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some(sum / n as f64)
    }
}

/// Hybrid probability (§11.3) with tte-dependent weighting. Early in the window
/// (tte_norm near 1.0) we lean on the parametric calibrated probability; late
/// in the window we lean more on retrieval.
pub fn p_hybrid(p_star: f64, p_nn: Option<f64>, tte_norm: f64) -> f64 {
    match p_nn {
        None => p_star,
        Some(p) => {
            // w_param ranges from ~1.0 early to ~0.5 late.
            let tte_norm = tte_norm.clamp(0.0, 1.0);
            let w_param = 0.5 + 0.5 * tte_norm;
            w_param * p_star + (1.0 - w_param) * p
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(ts: u64, v: &[f64], y: f64, family: &str) -> RetrievalItem {
        RetrievalItem {
            sequence_ts_ms: ts,
            vector: v.to_vec(),
            market_family: Some(family.into()),
            tick_regime: None,
            vol_regime: None,
            resolved_up: Some(y),
            row_id: None,
        }
    }

    #[test]
    fn brute_force_retrieves_closest() {
        let idx = BruteForceRetrieval::new(vec![
            item(1, &[0.0, 0.0], 1.0, "btc_5m"),
            item(2, &[1.0, 1.0], 0.0, "btc_5m"),
            item(3, &[0.2, 0.1], 1.0, "btc_5m"),
        ]);
        let q = RetrievalQuery {
            vector: vec![0.1, 0.1],
            k: 2,
            market_family: Some("btc_5m".into()),
            tick_regime: None,
            vol_regime: None,
            min_sequence_ts_ms: None,
            max_sequence_ts_ms: None,
        };
        let hits = idx.query(&q);
        assert_eq!(hits.len(), 2);
        // (0.2,0.1) is closer to (0.1,0.1) than (0,0) is.
        assert_eq!(hits[0].sequence_ts_ms, 3);
        let ids: std::collections::HashSet<u64> =
            hits.iter().map(|h| h.sequence_ts_ms).collect();
        assert!(ids.contains(&1) && ids.contains(&3));
    }

    #[test]
    fn p_hybrid_weights_shift_with_tte() {
        // Early in window (tte_norm=1.0) weight on parametric is 1.0.
        assert!((p_hybrid(0.7, Some(0.2), 1.0) - 0.7).abs() < 1e-9);
        // Late in window (tte_norm=0.0) weight splits 0.5/0.5.
        assert!((p_hybrid(0.7, Some(0.2), 0.0) - 0.45).abs() < 1e-9);
    }
}
