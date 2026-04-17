//! RAFT: Resolution-Aligned Fee-Aware Trading.
//!
//! Library entry point. The binary crate in `src/bin/raft.rs` wires these
//! modules together into a running service. Modules mirror the design doc
//! section numbers (§7.7, §16.1) as closely as possible.

pub mod config;
pub mod time_util;
pub mod types;

pub mod collector;
pub mod features;
pub mod models;
pub mod execution;
pub mod storage;

pub mod event_bus;
pub mod health;

pub use config::Config;
