pub mod api_admin;
pub mod api_ollama;
pub mod api_openai;
pub mod config;
pub mod discovery;
pub mod proxy;
pub mod registry;

use config::{Config, ServerConfig};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<registry::Registry>,
    pub http: reqwest::Client,
    pub state_file: PathBuf,
}

impl AppState {
    pub fn new(
        config: &Config,
        state_servers: Vec<ServerConfig>,
    ) -> Result<AppState, registry::RegistryError> {
        let mut servers: Vec<(ServerConfig, registry::Origin)> = config
            .server
            .iter()
            .cloned()
            .map(|s| (s, registry::Origin::Config))
            .collect();
        servers.extend(
            state_servers
                .into_iter()
                .map(|s| (s, registry::Origin::State)),
        );
        let registry = registry::Registry::new(servers)?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                config.router.request_timeout_secs,
            ))
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .expect("reqwest client builder failed");
        Ok(AppState {
            registry: Arc::new(registry),
            http,
            state_file: config.router.state_file.clone(),
        })
    }
}

pub fn build_router(state: AppState) -> axum::Router {
    axum::Router::new()
        .merge(api_openai::router())
        .merge(api_ollama::router())
        .merge(api_admin::router())
        .with_state(state)
}
