//! Collectors (§7.7). Each collector owns a single external feed, normalizes
//! incoming messages, and publishes to the event bus.

pub mod backoff;
pub mod rtds_ws;
pub mod market_ws;
pub mod rest_hydrator;
