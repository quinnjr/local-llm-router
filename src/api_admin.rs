use crate::config::{ServerConfig, StateFile};
use crate::registry::{Origin, RegistryError, exposed_id};
use crate::{AppState, discovery};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/admin/servers", get(list_servers).post(add_server))
        .route("/admin/servers/{name}", delete(remove_server))
        .route("/admin/refresh", post(refresh))
}

fn server_json(entry: &crate::registry::ServerEntry) -> serde_json::Value {
    let models: Vec<String> = entry
        .models
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|m| exposed_id(&entry.config.name, &m.upstream_name))
        .collect();
    serde_json::json!({
        "name": entry.config.name,
        "url": entry.config.url,
        "flavor": entry.config.flavor.to_string(),
        "origin": match entry.origin {
            Origin::Config => "config",
            Origin::State => "state",
        },
        "reachable": entry.reachable(),
        "models": models,
    })
}

async fn list_servers(State(state): State<AppState>) -> Response {
    let servers: Vec<serde_json::Value> = state
        .registry
        .list_servers()
        .iter()
        .map(server_json)
        .collect();
    Json(serde_json::json!({"servers": servers})).into_response()
}

fn persist(state: &AppState) -> Result<(), Response> {
    // state_servers() iterates a HashMap; sort so state.toml doesn't flap between runs.
    let mut servers = state.registry.state_servers();
    servers.sort_by(|a, b| a.name.cmp(&b.name));
    StateFile::save(&state.state_file, &servers).map_err(|e| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to persist state file: {e}"),
        )
    })
}

async fn add_server(
    State(state): State<AppState>,
    Json(config): Json<ServerConfig>,
) -> Response {
    if config.name.is_empty() || config.name.contains('/') {
        return error(
            StatusCode::CONFLICT,
            "server name must not be empty or contain '/'",
        );
    }
    if reqwest::Url::parse(&config.url).is_err() {
        return error(StatusCode::BAD_REQUEST, "invalid url");
    }
    let name = config.name.clone();
    if let Err(e) = state.registry.upsert_server(config, Origin::State) {
        let status = match e {
            RegistryError::DuplicateName(_) | RegistryError::InvalidName(_) => {
                StatusCode::CONFLICT
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        return error(status, &e.to_string());
    }
    if let Err(resp) = persist(&state) {
        let _ = state.registry.remove_server(&name);
        return resp;
    }
    discovery::refresh_one(&state, &name).await;
    let Some(entry) = state.registry.get_server(&name) else {
        return error(
            StatusCode::CONFLICT,
            "server was removed concurrently",
        );
    };
    (StatusCode::CREATED, Json(server_json(&entry))).into_response()
}

async fn remove_server(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    if let Err(e) = state.registry.remove_server(&name) {
        let status = match e {
            RegistryError::UnknownServer(_) => StatusCode::NOT_FOUND,
            RegistryError::ConfigOrigin(_) => StatusCode::CONFLICT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        return error(status, &e.to_string());
    }
    if let Err(resp) = persist(&state) {
        return resp;
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn refresh(State(state): State<AppState>) -> Response {
    let results = discovery::refresh_all(&state).await;
    let results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|(name, reachable)| serde_json::json!({"name": name, "reachable": reachable}))
        .collect();
    Json(serde_json::json!({"results": results})).into_response()
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": message}))).into_response()
}
