//! Feature builder (§10, Appendix C).
//!
//! Maintains per-market live state and per-reference live state from the event
//! bus, then emits `StateFeatures` snapshots on demand (every
//! `strategy.state_interval_ms`). Core state vector per §10.1.

pub mod live_state;
pub mod vol;
pub mod flow;
pub mod state_builder;

pub use live_state::LiveStateStore;
pub use state_builder::{build_state_features, StateFeatures};
