//! Execution layer (§13, §16.1).
//!
//! Split:
//!   * `router`        — decides no-trade / maker / taker, emits OrderIntent
//!   * `quote_manager` — manages live maker quotes (post-only, cancel/requote)
//!   * `risk`          — position/loss/session-limit gates
//!   * `heartbeat`     — liveness watchdog that trips kill-switches
//!
//! Broker connectivity (CLOB signed-order placement) is intentionally behind a
//! `Broker` trait so the paper-trader and live-trader share the same policy
//! code.

pub mod broker;
pub mod cost_model;
pub mod heartbeat;
pub mod quote_manager;
pub mod risk;
pub mod router;

pub use broker::{Broker, BrokerEvent, OrderIntent, PaperBroker};
pub use cost_model::AllInCost;
pub use heartbeat::Heartbeat;
pub use risk::{RiskConfig, RiskManager};
pub use router::{Decision, RouteContext, Router};
