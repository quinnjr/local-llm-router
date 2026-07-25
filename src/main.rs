use local_llm_router::config::{Config, StateFile};
use local_llm_router::{AppState, build_router, discovery};
use std::path::PathBuf;
use std::process::exit;

fn parse_config_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut path = PathBuf::from("./config.toml");
    while let Some(arg) = args.next() {
        if arg == "--config" {
            match args.next() {
                Some(p) => path = PathBuf::from(p),
                None => {
                    eprintln!("--config requires a path argument");
                    exit(2);
                }
            }
        } else if !arg.starts_with('-') {
            path = PathBuf::from(arg);
        }
    }
    path
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config_path = parse_config_path();
    let config = match Config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load {}: {e}", config_path.display());
            exit(1);
        }
    };
    let state_file = match StateFile::load(&config.router.state_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "failed to load state file {}: {e}",
                config.router.state_file.display()
            );
            exit(1);
        }
    };
    let state = match AppState::new(&config, state_file.server) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("invalid server configuration: {e}");
            exit(1);
        }
    };

    let results = discovery::refresh_all(&state).await;
    for (name, reachable) in &results {
        if !reachable {
            tracing::warn!(server = %name, "server unreachable at startup; will retry on refresh");
        }
    }

    let app = build_router(state);
    let listener = match tokio::net::TcpListener::bind(&config.router.listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {}: {e}", config.router.listen);
            exit(1);
        }
    };
    tracing::info!(listen = %config.router.listen, "local-llm-router up");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("server error: {e}");
        exit(1);
    }
}
