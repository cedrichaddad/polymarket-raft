//! Parquet sink (§7.10).
//!
//! Layout:
//!   data/raw/source={rtds,market_ws,trades,meta,orders,fills}/date=YYYY-MM-DD/<id>.parquet
//!
//! Strategy:
//!   * One writer per source, holding a RecordBatch buffer.
//!   * Flush & roll when buffer reaches `roll_max_mb` *or* `roll_interval_secs`.
//!   * On close, sync and rename <id>.parquet.tmp -> <id>.parquet.
//!
//! v1 writes compact, schema-stable columns. Richer event bodies are persisted
//! as stringified JSON in `body_json` so we don't lose information when the
//! upstream schema evolves.

use crate::event_bus::EventReceiver;
use crate::time_util::now_ms;
use crate::types::ExternalEvent;

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, Float64Builder, StringBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use chrono::{TimeZone, Utc};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::broadcast::error::RecvError;
use tracing::{debug, error, info, warn};

const SOURCE_RTDS: &str = "rtds";
const SOURCE_MARKET: &str = "market_ws";
const SOURCE_TRADES: &str = "trades";
const SOURCE_META: &str = "meta";
const SOURCE_ORDERS: &str = "orders";
const SOURCE_FILLS: &str = "fills";

pub struct ParquetSink {
    root: PathBuf,
    roll_interval_secs: u64,
    roll_max_bytes: u64,
    /// Per-source in-progress batches.
    batches: HashMap<&'static str, Batch>,
}

struct Batch {
    schema: Arc<Schema>,
    // Columnar builders, keyed by column name.
    cols: Columns,
    rows: usize,
    approx_bytes: u64,
    opened_ms: u64,
    date_key: String,
    file_id: String,
}

struct Columns {
    ts_exchange_ms: UInt64Builder,  // 0 => null encoded upstream if needed
    ts_recv_local_ms: UInt64Builder,
    f1: Float64Builder,
    f2: Float64Builder,
    s1: StringBuilder,
    s2: StringBuilder,
    s3: StringBuilder,
    body_json: StringBuilder,
}

impl ParquetSink {
    pub fn new(root: impl Into<PathBuf>, roll_interval_secs: u64, roll_max_mb: u64) -> Self {
        Self {
            root: root.into(),
            roll_interval_secs,
            roll_max_bytes: roll_max_mb.saturating_mul(1024 * 1024),
            batches: HashMap::new(),
        }
    }

    /// Run the sink, consuming events from the bus until the channel closes.
    pub async fn run(mut self, mut rx: EventReceiver) -> Result<()> {
        info!(target: "sink", root = %self.root.display(), "parquet sink starting");
        fs::create_dir_all(&self.root)?;

        let mut roll_timer = tokio::time::interval(std::time::Duration::from_secs(
            self.roll_interval_secs.max(1),
        ));
        roll_timer.tick().await; // consume immediate tick

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Ok(ev) => {
                            if let Err(e) = self.append(&ev) {
                                warn!(target: "sink", error = %e, "append failed");
                            }
                        }
                        Err(RecvError::Lagged(n)) => {
                            warn!(target: "sink", lagged = n, "broadcast lagged");
                        }
                        Err(RecvError::Closed) => {
                            info!(target: "sink", "bus closed, flushing");
                            let _ = self.flush_all();
                            return Ok(());
                        }
                    }
                }
                _ = roll_timer.tick() => {
                    if let Err(e) = self.flush_rolling() {
                        warn!(target: "sink", error = %e, "roll flush failed");
                    }
                }
            }
        }
    }

    fn append(&mut self, ev: &ExternalEvent) -> Result<()> {
        let (src, cols) = event_to_columns(ev);
        let approx = estimate_bytes(&cols);
        let batch = self
            .batches
            .entry(src)
            .or_insert_with(|| Batch::new(src, now_ms()));
        batch.push(cols);
        batch.approx_bytes += approx;

        // Roll by size.
        if batch.approx_bytes >= self.roll_max_bytes {
            let src_static = src;
            self.roll_source(src_static)?;
        }
        Ok(())
    }

    fn flush_rolling(&mut self) -> Result<()> {
        let now = now_ms();
        let keys: Vec<&'static str> = self
            .batches
            .iter()
            .filter(|(_, b)| {
                b.rows > 0
                    && now.saturating_sub(b.opened_ms) >= self.roll_interval_secs * 1000
            })
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            self.roll_source(k)?;
        }
        Ok(())
    }

    fn flush_all(&mut self) -> Result<()> {
        let keys: Vec<&'static str> = self.batches.keys().copied().collect();
        for k in keys {
            self.roll_source(k)?;
        }
        Ok(())
    }

    fn roll_source(&mut self, src: &'static str) -> Result<()> {
        let Some(mut batch) = self.batches.remove(src) else {
            return Ok(());
        };
        if batch.rows == 0 {
            return Ok(());
        }
        let record = batch.finish()?;
        let partition_dir = self.root.join(format!("source={src}")).join(format!("date={}", batch.date_key));
        fs::create_dir_all(&partition_dir)?;
        let final_path = partition_dir.join(format!("{}.parquet", batch.file_id));
        let tmp_path = partition_dir.join(format!("{}.parquet.tmp", batch.file_id));

        let file = File::create(&tmp_path).with_context(|| format!("create {tmp_path:?}"))?;
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut writer = ArrowWriter::try_new(file, record.schema(), Some(props))?;
        writer.write(&record)?;
        writer.close()?;

        // fsync the tmp then rename (§7.10.2).
        if let Ok(f) = OpenOptions::new().read(true).write(true).open(&tmp_path) {
            let _ = f.sync_all();
        }
        fs::rename(&tmp_path, &final_path).with_context(|| format!("rename {final_path:?}"))?;
        debug!(target: "sink", src = src, rows = record.num_rows(), path = %final_path.display(), "rolled");
        Ok(())
    }
}

impl Batch {
    fn new(src: &'static str, opened_ms: u64) -> Self {
        let schema = schema_for(src);
        let date_key = date_key_for(opened_ms);
        let file_id = format!("{opened_ms}-{}", &uuid::Uuid::new_v4().to_string()[..8]);
        Self {
            schema,
            cols: Columns::new(),
            rows: 0,
            approx_bytes: 0,
            opened_ms,
            date_key,
            file_id,
        }
    }

    fn push(&mut self, row: EventColumns) {
        self.cols.ts_exchange_ms.append_value(row.ts_exchange_ms.unwrap_or(0));
        self.cols.ts_recv_local_ms.append_value(row.ts_recv_local_ms);
        match row.f1 {
            Some(x) => self.cols.f1.append_value(x),
            None => self.cols.f1.append_null(),
        }
        match row.f2 {
            Some(x) => self.cols.f2.append_value(x),
            None => self.cols.f2.append_null(),
        }
        self.cols.s1.append_value(row.s1.as_deref().unwrap_or(""));
        self.cols.s2.append_value(row.s2.as_deref().unwrap_or(""));
        self.cols.s3.append_value(row.s3.as_deref().unwrap_or(""));
        self.cols.body_json.append_value(row.body_json.as_deref().unwrap_or(""));
        self.rows += 1;
    }

    fn finish(&mut self) -> Result<RecordBatch> {
        let arrays: Vec<ArrayRef> = vec![
            Arc::new(self.cols.ts_exchange_ms.finish()),
            Arc::new(self.cols.ts_recv_local_ms.finish()),
            Arc::new(self.cols.f1.finish()),
            Arc::new(self.cols.f2.finish()),
            Arc::new(self.cols.s1.finish()),
            Arc::new(self.cols.s2.finish()),
            Arc::new(self.cols.s3.finish()),
            Arc::new(self.cols.body_json.finish()),
        ];
        let rb = RecordBatch::try_new(self.schema.clone(), arrays)?;
        Ok(rb)
    }
}

impl Columns {
    fn new() -> Self {
        Self {
            ts_exchange_ms: UInt64Builder::new(),
            ts_recv_local_ms: UInt64Builder::new(),
            f1: Float64Builder::new(),
            f2: Float64Builder::new(),
            s1: StringBuilder::new(),
            s2: StringBuilder::new(),
            s3: StringBuilder::new(),
            body_json: StringBuilder::new(),
        }
    }
}

/// Unified row layout. Per-source meaning of the generic columns is documented
/// here and mirrored in `schema_for` and `event_to_columns`.
///
///  rtds       : s1=topic, s2=symbol, f1=value, f2=None, s3=None
///  market_ws  : s1=market_id, s2=asset_id, s3=event_type, f1=best_bid, f2=best_ask
///  trades     : s1=market_id, s2=asset_id, s3=side_aggressor, f1=price, f2=size
///  meta       : s1=market_id, s2=window_type, s3=asset_yes_id, f1=start_ts, f2=end_ts
///  orders     : s1=local_id, s2=market_id, s3=side, f1=price, f2=size
///  fills      : s1=exchange_order_id, s2=market_id, s3=maker_or_taker, f1=fill_price, f2=fill_size
struct EventColumns {
    ts_exchange_ms: Option<u64>,
    ts_recv_local_ms: u64,
    f1: Option<f64>,
    f2: Option<f64>,
    s1: Option<String>,
    s2: Option<String>,
    s3: Option<String>,
    body_json: Option<String>,
}

fn schema_for(_src: &str) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("ts_exchange_ms", DataType::UInt64, false),
        Field::new("ts_recv_local_ms", DataType::UInt64, false),
        Field::new("f1", DataType::Float64, true),
        Field::new("f2", DataType::Float64, true),
        Field::new("s1", DataType::Utf8, true),
        Field::new("s2", DataType::Utf8, true),
        Field::new("s3", DataType::Utf8, true),
        Field::new("body_json", DataType::Utf8, true),
    ]))
}

fn event_to_columns(ev: &ExternalEvent) -> (&'static str, EventColumns) {
    match ev {
        ExternalEvent::RtdsCrypto(u) => (
            SOURCE_RTDS,
            EventColumns {
                ts_exchange_ms: Some(u.ts_payload_ms),
                ts_recv_local_ms: u.ts_recv_local_ms,
                f1: Some(u.value),
                f2: None,
                s1: Some(u.topic.as_str().to_string()),
                s2: Some(u.symbol.clone()),
                s3: None,
                body_json: None,
            },
        ),
        ExternalEvent::MarketBook(b) => (
            SOURCE_MARKET,
            EventColumns {
                ts_exchange_ms: b.ts_exchange_ms,
                ts_recv_local_ms: b.ts_recv_local_ms,
                f1: b.best_bid,
                f2: b.best_ask,
                s1: Some(b.market_id.clone()),
                s2: Some(b.asset_id.clone()),
                s3: Some(b.event_type.clone()),
                body_json: Some(b.book_json.clone()),
            },
        ),
        ExternalEvent::MarketTrade(t) => (
            SOURCE_TRADES,
            EventColumns {
                ts_exchange_ms: t.ts_exchange_ms,
                ts_recv_local_ms: t.ts_recv_local_ms,
                f1: Some(t.price),
                f2: Some(t.size),
                s1: Some(t.market_id.clone()),
                s2: Some(t.asset_id.clone()),
                s3: t.side_aggressor.map(|s| format!("{s:?}")),
                body_json: t.trade_id.clone(),
            },
        ),
        ExternalEvent::MarketMeta(m) => (
            SOURCE_META,
            EventColumns {
                ts_exchange_ms: Some(m.start_ts_ms),
                ts_recv_local_ms: now_ms(),
                f1: Some(m.start_ts_ms as f64),
                f2: Some(m.end_ts_ms as f64),
                s1: Some(m.market_id.clone()),
                s2: Some(m.window_type.clone()),
                s3: Some(m.asset_yes_id.clone()),
                body_json: Some(m.fee_schedule_json.clone()),
            },
        ),
        ExternalEvent::OrderAck(o) => (
            SOURCE_ORDERS,
            EventColumns {
                ts_exchange_ms: Some(o.ts_recv_local_ms),
                ts_recv_local_ms: o.ts_recv_local_ms,
                f1: Some(o.price),
                f2: Some(o.size),
                s1: Some(o.local_order_id.clone()),
                s2: Some(o.market_id.clone()),
                s3: Some(format!("{:?}", o.side)),
                body_json: Some(o.status.clone()),
            },
        ),
        ExternalEvent::Fill(f) => (
            SOURCE_FILLS,
            EventColumns {
                ts_exchange_ms: Some(f.fill_ts_ms),
                ts_recv_local_ms: f.fill_ts_ms,
                f1: Some(f.fill_price),
                f2: Some(f.fill_size),
                s1: Some(f.exchange_order_id.clone()),
                s2: Some(f.market_id.clone()),
                s3: Some(f.maker_or_taker.clone()),
                body_json: Some(format!(
                    "{{\"fee\":{},\"rebate\":{}}}",
                    f.estimated_fee_usdc, f.estimated_rebate_usdc
                )),
            },
        ),
    }
}

fn estimate_bytes(row: &EventColumns) -> u64 {
    let s = row
        .s1
        .as_ref()
        .map(|x| x.len())
        .unwrap_or(0)
        + row.s2.as_ref().map(|x| x.len()).unwrap_or(0)
        + row.s3.as_ref().map(|x| x.len()).unwrap_or(0)
        + row.body_json.as_ref().map(|x| x.len()).unwrap_or(0);
    (s + 48) as u64 // plus fixed-size numeric columns
}

fn date_key_for(ms: u64) -> String {
    let dt = Utc.timestamp_millis_opt(ms as i64).single().unwrap_or_else(|| Utc::now());
    dt.format("%Y-%m-%d").to_string()
}

#[allow(dead_code)]
fn ensure_parent(p: &Path) -> Result<()> {
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

// Keep `error!` import satisfied if/when used.
#[allow(dead_code)]
fn _suppress_unused() {
    error!("");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RtdsTopic;

    #[test]
    fn date_key_formats_utc() {
        let s = date_key_for(0);
        assert_eq!(s, "1970-01-01");
    }

    #[test]
    fn event_to_columns_handles_rtds() {
        let ev = ExternalEvent::RtdsCrypto(crate::types::RtdsCryptoUpdate {
            topic: RtdsTopic::Binance,
            symbol: "btcusdt".into(),
            ts_server_ms: 10,
            ts_payload_ms: 9,
            ts_recv_local_ms: 11,
            value: 100_000.0,
        });
        let (src, cols) = event_to_columns(&ev);
        assert_eq!(src, SOURCE_RTDS);
        assert_eq!(cols.f1, Some(100_000.0));
        assert_eq!(cols.s1.as_deref(), Some("binance"));
    }
}
