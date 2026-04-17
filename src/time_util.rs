//! Time helpers. Every event must carry ts_exchange_ms/ts_payload_ms,
//! ts_server_ms, and ts_recv_local_ms (§7.9).

use std::time::{SystemTime, UNIX_EPOCH};

#[inline]
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
