//! `raft` — main service binary.
//!
//! Wires the Appendix F build order into a runnable process:
//!   1. load config
//!   2. start RTDS + market-ws + REST-hydrator collectors
//!   3. fan events through the bus to the Parquet sink and LiveStateStore
//!   4. (paper/live modes only) run the decision loop
//!
//! Modes per §7.12:
//!   research — collect and persist only
//!   paper    — collect + decisions against a PaperBroker
//!   live     — same plus real broker (not yet wired; v1 guards against this)

use clap::{Parser, Subcommand};
use std::sync::Arc;

use raft::collector::market_ws::MarketWsClient;
use raft::collector::rest_hydrator::RestHydrator;
use raft::collector::rtds_ws::RtdsClient;
use raft::config::{Config, RunMode};
use raft::event_bus::create_bus;
use raft::execution::{AllInCost, Heartbeat, PaperBroker};
use raft::execution::broker::PaperFees;
use raft::execution::quote_manager::QuoteManager;
use raft::execution::risk::{RiskConfig, RiskManager};
use raft::execution::router::{Decision, RouteContext, Router};
use raft::features::state_builder::build_state_features;
use raft::features::LiveStateStore;
use raft::health::HealthMonitor;
use raft::models::{fair_prob_backbone, FairValueParams, IsotonicCalibrator, Calibrator};
use raft::models::retrieval::{p_hybrid, p_nn, BruteForceRetrieval, Retrieval, RetrievalQuery};
use raft::storage::ParquetSink;

#[derive(Parser, Debug)]
#[command(name = "raft", about = "Resolution-Aligned Fee-Aware Trading", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the service (collectors + sink; decisions if paper/live).
    Run {
        #[arg(long, default_value = "config/raft.yaml")]
        config: String,
        /// Override mode from config (research | paper | live).
        #[arg(long)]
        mode: Option<String>,
    },
    /// Print resolved config for sanity-checking.
    PrintConfig {
        #[arg(long, default_value = "config/raft.yaml")]
        config: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.cmd {
        Command::PrintConfig { config } => {
            let cfg = Config::load_from_file(&config)?;
            println!("{}", serde_yaml::to_string(&cfg)?);
            Ok(())
        }
        Command::Run { config, mode } => {
            let mut cfg = Config::load_from_file(&config)?;
            if let Some(m) = mode {
                cfg.mode = match m.as_str() {
                    "research" => RunMode::Research,
                    "paper" => RunMode::Paper,
                    "live" => RunMode::Live,
                    other => anyhow::bail!("unknown mode: {other}"),
                };
            }
            run(cfg).await
        }
    }
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).with_target(true).init();
}

async fn run(cfg: Config) -> anyhow::Result<()> {
    tracing::info!(mode = ?cfg.mode, "starting raft");
    if matches!(cfg.mode, RunMode::Live) {
        anyhow::bail!("live mode requires a real broker implementation not present in v1");
    }

    let (tx, rx_sink) = create_bus(65_536);
    let health = Arc::new(HealthMonitor::new(
        cfg.action_gates.binance_stale_ms,
        cfg.action_gates.chainlink_stale_ms,
    ));
    let store = Arc::new(LiveStateStore::new());

    // Parquet sink task.
    let sink = ParquetSink::new(&cfg.storage.parquet_root, cfg.storage.roll_interval_secs, cfg.storage.roll_max_mb);
    let sink_task = tokio::spawn(async move { sink.run(rx_sink).await });

    // Live-state consumer task.
    let mut rx_state = tx.subscribe();
    let store_for_state = store.clone();
    let state_task = tokio::spawn(async move {
        loop {
            match rx_state.recv().await {
                Ok(ev) => store_for_state.apply(&ev),
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(lagged = n, "state consumer lagged");
                }
                Err(_) => return,
            }
        }
    });

    // RTDS collector.
    let rtds = Arc::new(RtdsClient::new(cfg.rtds.clone(), tx.clone(), health.clone()));
    let rtds_task = tokio::spawn(async move { rtds.run().await });

    // REST hydrator + market-ws collector (if enabled).
    let hydrator = Arc::new(RestHydrator::new(cfg.hydration.clone(), tx.clone()));
    // Prime the hydrator so assets exist before market-ws connects.
    if let Err(e) = hydrator.refresh_once().await {
        tracing::warn!(error = %e, "initial hydration failed — continuing");
    }
    let hydrator_loop = hydrator.clone();
    let hydrator_task = tokio::spawn(async move { hydrator_loop.run().await });

    let market_client = if cfg.market_ws.enabled {
        let client = Arc::new(MarketWsClient::new(cfg.market_ws.clone(), tx.clone()));
        client.set_assets(hydrator.current_assets().await).await;
        let c = client.clone();
        tokio::spawn(async move { c.run().await });
        Some(client)
    } else {
        None
    };

    // Push fresh assets to the market-ws client periodically.
    if let Some(c) = market_client.clone() {
        let hyd = hydrator.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                c.set_assets(hyd.current_assets().await).await;
            }
        });
    }

    // Decision loop only in paper mode.
    if matches!(cfg.mode, RunMode::Paper) {
        spawn_decision_loop(cfg.clone(), store.clone(), health.clone());
    }

    tokio::signal::ctrl_c().await.ok();
    tracing::info!("shutting down");

    // Drop the bus sender by shadowing; downstream tasks will exit.
    drop(tx);

    // Await tasks that flush cleanly (sink, state). Others are fire-and-forget.
    let _ = tokio::join!(sink_task, state_task);
    let _ = rtds_task.abort();
    let _ = hydrator_task.abort();
    Ok(())
}

fn spawn_decision_loop(
    cfg: Config,
    store: Arc<LiveStateStore>,
    health: Arc<HealthMonitor>,
) {
    tokio::spawn(async move {
        let broker = PaperBroker::new(PaperFees::default());
        let qm = Arc::new(QuoteManager::new(broker.clone()));
        let risk = Arc::new(RiskManager::new(RiskConfig::default()));
        let hb = Arc::new(Heartbeat::new(5_000));
        let calibrator = IsotonicCalibrator::identity();
        let retrieval: Arc<dyn Retrieval> = Arc::new(BruteForceRetrieval::new(Vec::new()));

        let horizon_seconds = match cfg.strategy.market_family.as_str() {
            "btc_15m" => 900.0,
            _ => 300.0,
        };

        let tick = std::time::Duration::from_millis(cfg.strategy.state_interval_ms.max(100));
        let mut interval = tokio::time::interval(tick);
        loop {
            interval.tick().await;
            hb.tick();
            if let Some(r) = hb.check() {
                tracing::error!(reason = %r, "heartbeat tripped");
                qm.force_cancel_all();
                risk.kill("heartbeat");
                return;
            }
            // Snapshot markets and health.
            let snapshot_health = health.snapshot();
            // In v1 the "opening reference" is approximated by the most recent
            // Chainlink print — a truthful implementation snapshots the
            // Chainlink print at window start and persists it. Upgrading this
            // is Phase 1 research work.
            let market_ids: Vec<String> = store.markets.iter().map(|e| e.key().clone()).collect();
            for market_id in market_ids {
                let Some(features) = build_state_features(&store, &market_id, &|meta| {
                    // Simple placeholder: snapshot chainlink at first sighting.
                    if meta.start_ts_ms == 0 {
                        return store.reference.read().chainlink_price;
                    }
                    store.reference.read().chainlink_price
                }, horizon_seconds) else { continue };

                // Parametric p_0 using spec defaults + rv_30s as sigma_per_sec.
                let params = FairValueParams {
                    drift: 0.0,
                    sigma_per_sec: features.rv_30s.max(1e-6),
                };
                let tau_s = features.tte_ms as f64 / 1000.0;
                let Some(p_0) = fair_prob_backbone(features.chainlink_price, features.open_ref_price, tau_s, params) else { continue };
                let p_star = calibrator.apply(p_0);

                let query = RetrievalQuery {
                    vector: features.as_vector().to_vec(),
                    k: cfg.strategy.retrieval_k,
                    market_family: Some(cfg.strategy.market_family.clone()),
                    tick_regime: None,
                    vol_regime: None,
                    min_sequence_ts_ms: None,
                    max_sequence_ts_ms: None,
                };
                let hits = retrieval.query(&query);
                let p_nn_val = p_nn(&hits);
                let p_hyb = p_hybrid(p_star, p_nn_val, features.tte_norm);

                let ctx = RouteContext {
                    features: &features,
                    p_hybrid: p_hyb,
                    p_market: features.market_prob,
                    all_in_cost: AllInCost::default(),
                    health: &snapshot_health,
                    center_low: cfg.strategy.center_prob_low,
                    center_high: cfg.strategy.center_prob_high,
                    min_spread_for_maker: cfg.strategy.min_spread_ticks_for_maker as f64 * 0.01,
                    taker_window_secs: 30.0,
                    expected_adverse_prob: 0.004,
                    expected_rebate_prob: 0.0,
                };
                let decision = Router::decide(&ctx);
                tracing::debug!(
                    market = %market_id,
                    p_0,
                    p_star,
                    p_nn = ?p_nn_val,
                    p_hyb,
                    p_mkt = features.market_prob,
                    decision = ?decision,
                    "decision"
                );
                // In paper mode wire maker decisions through the quote manager.
                // Taker crosses would go through the broker directly; we log-only
                // in v1 because the paper broker fills instantly which would skew
                // research numbers — defer to the Python backtester.
                if let Some(meta) = store.metadata.get(&market_id) {
                    match decision {
                        Decision::MakerQuote { .. } => {
                            let check = risk.check_new_order(&market_id, &cfg.strategy.market_family, broker.open_orders().len() as f64, 0.0);
                            if matches!(check, raft::execution::risk::RiskVerdict::Ok) {
                                qm.apply(&market_id, &meta.asset_yes_id, &decision);
                            }
                        }
                        _ => {
                            qm.apply(&market_id, &meta.asset_yes_id, &decision);
                        }
                    }
                }
            }
        }
    });
}
