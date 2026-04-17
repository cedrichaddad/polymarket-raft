//! Event bus (§7.7.5). Fans normalized events out to:
//!  - parquet sink
//!  - feature builder
//!  - live execution service
//!
//! Implemented as a `tokio::sync::broadcast` channel so multiple consumers
//! receive every event. Consumers that fall behind will see `Lagged` errors
//! and should treat that as a health fault.

use crate::types::ExternalEvent;
use tokio::sync::broadcast;

pub type EventSender = broadcast::Sender<ExternalEvent>;
pub type EventReceiver = broadcast::Receiver<ExternalEvent>;

pub fn create_bus(capacity: usize) -> (EventSender, EventReceiver) {
    broadcast::channel(capacity)
}
