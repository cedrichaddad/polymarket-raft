//! Configuration loaded from YAML per Appendix B of the design doc.

use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_mode")]
    pub mode: RunMode,
    pub rtds: RtdsConfig,
    pub market_ws: MarketWsConfig,
    pub hydration: HydrationConfig,
    pub storage: StorageConfig,
    pub action_gates: ActionGates,
    pub strategy: StrategyConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum RunMode {
    Research,
    Paper,
    Live,
}

fn default_mode() -> RunMode {
    RunMode::Research
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtdsConfig {
    pub url: String,
    #[serde(default = "default_ping_ms")]
    pub ping_interval_ms: u64,
    #[serde(default)]
    pub reconnect: BackoffConfig,
    pub subscriptions: Vec<RtdsSubscriptionConfig>,
}

fn default_ping_ms() -> u64 {
    5000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtdsSubscriptionConfig {
    pub topic: String,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub filters: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketWsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_initial_backoff_ms")]
    pub reconnect_initial_backoff_ms: u64,
    #[serde(default = "default_max_backoff_ms")]
    pub reconnect_max_backoff_ms: u64,
    #[serde(default = "default_market_ws_url")]
    pub url: String,
}

fn default_market_ws_url() -> String {
    "wss://ws-subscriptions-clob.polymarket.com/ws/market".to_string()
}

fn default_true() -> bool {
    true
}

fn default_initial_backoff_ms() -> u64 {
    250
}

fn default_max_backoff_ms() -> u64 {
    5000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackoffConfig {
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
    #[serde(default = "default_true")]
    pub jitter: bool,
}

impl Default for BackoffConfig {
    fn default() -> Self {
        Self {
            initial_backoff_ms: 250,
            max_backoff_ms: 5000,
            jitter: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrationConfig {
    #[serde(default = "default_refresh_secs")]
    pub refresh_secs: u64,
    #[serde(default = "default_gamma_url")]
    pub gamma_url: String,
}

fn default_refresh_secs() -> u64 {
    60
}

fn default_gamma_url() -> String {
    "https://gamma-api.polymarket.com".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    pub parquet_root: String,
    #[serde(default = "default_roll_secs")]
    pub roll_interval_secs: u64,
    #[serde(default = "default_roll_mb")]
    pub roll_max_mb: u64,
}

fn default_roll_secs() -> u64 {
    300
}

fn default_roll_mb() -> u64 {
    64
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionGates {
    #[serde(default = "default_binance_stale_ms")]
    pub binance_stale_ms: u64,
    #[serde(default = "default_chainlink_stale_ms")]
    pub chainlink_stale_ms: u64,
    #[serde(default = "default_true")]
    pub disable_taker_if_chainlink_stale: bool,
    #[serde(default = "default_true")]
    pub disable_all_if_both_stale: bool,
}

fn default_binance_stale_ms() -> u64 {
    2000
}

fn default_chainlink_stale_ms() -> u64 {
    5000
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    #[serde(default = "default_market_family")]
    pub market_family: String,
    #[serde(default = "default_state_interval")]
    pub state_interval_ms: u64,
    #[serde(default = "default_k")]
    pub retrieval_k: usize,
    #[serde(default = "default_ef")]
    pub retrieval_ef: usize,
    #[serde(default = "default_center_low")]
    pub center_prob_low: f64,
    #[serde(default = "default_center_high")]
    pub center_prob_high: f64,
    #[serde(default = "default_min_spread_ticks")]
    pub min_spread_ticks_for_maker: u32,
}

fn default_market_family() -> String {
    "btc_5m".to_string()
}
fn default_state_interval() -> u64 {
    1000
}
fn default_k() -> usize {
    64
}
fn default_ef() -> usize {
    128
}
fn default_center_low() -> f64 {
    0.40
}
fn default_center_high() -> f64 {
    0.60
}
fn default_min_spread_ticks() -> u32 {
    2
}

impl Config {
    pub fn load_from_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Config = serde_yaml::from_str(&text)?;
        Ok(cfg)
    }
}
