//! `ov-gateway-rs CONFIG.json`: serve the prototype gateway.
//!
//! Config: {"listen", "openviking_url", "storage_path", "upstream_base", "upstream_key",
//!          "keys": {"<gateway key>": {"account", "user_id", "openviking_key"}}}
//! The "default" policy is written to management.sqlite3 at start and read per request.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use ov_gateway_rs::canon::digest;
use ov_gateway_rs::kernel::{self, Credential, Ctx};
use ov_gateway_rs::ov::Viking;
use ov_gateway_rs::proxy;
use ov_gateway_rs::store::Store;
use serde_json::Value;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: ov-gateway-rs CONFIG.json");
    let config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let workers: usize = std::env::var("OVGW_WORKERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .max_blocking_threads(4)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(serve(config));
}

async fn serve(config: Value) {
    let s = |k: &str| {
        config[k]
            .as_str()
            .unwrap_or_else(|| panic!("config.{k}"))
            .to_string()
    };
    let store = Arc::new(Store::open(std::path::Path::new(&s("storage_path"))));
    let mut credentials = HashMap::new();
    for (key, v) in config["keys"].as_object().unwrap() {
        let account = v["account"].as_str().unwrap().to_string();
        credentials.insert(
            key.clone(),
            Credential {
                id: digest(key),
                account: account.clone(),
                user_id: v["user_id"].as_str().unwrap().to_string(),
                openviking_key: v["openviking_key"].as_str().unwrap().to_string(),
                policy_id: "default".into(),
                upstream_id: "mock-anthropic".into(),
            },
        );
        store.save_object(&account, "policies", "default", &kernel::default_policy());
    }
    let http = reqwest::Client::builder()
        .no_proxy()
        .pool_max_idle_per_host(4096)
        .tcp_nodelay(true)
        .build()
        .unwrap();
    let ctx = Arc::new(Ctx {
        store,
        viking: Viking::new(http.clone(), &s("openviking_url")),
        http,
        upstream_base: s("upstream_base").trim_end_matches('/').to_string(),
        upstream_key: s("upstream_key"),
        credentials,
    });
    tokio::spawn(kernel::capture_worker(ctx.clone()));
    let app = Router::new()
        .route("/health", get(proxy::health))
        .route("/v1/messages", post(proxy::messages))
        .route("/gateway/v1/messages", post(proxy::messages))
        .layer(DefaultBodyLimit::max(32 * 1024 * 1024))
        .with_state(ctx);
    let listener = tokio::net::TcpListener::bind(s("listen")).await.unwrap();
    eprintln!("ov-gateway-rs listening on {}", s("listen"));
    axum::serve(listener, app).await.unwrap();
}
