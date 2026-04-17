//! Local persistence (§7.10, §9.1).
//!
//! v1 uses partitioned Parquet files grouped by source family. The writer
//! owns file rolling (time/size) and atomic fsync-on-close. Downstream
//! research (DuckDB/Python) reads these files directly — they are the
//! canonical raw archive.

pub mod parquet_sink;

pub use parquet_sink::ParquetSink;
