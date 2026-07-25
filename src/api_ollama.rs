use crate::AppState;
use crate::config::Flavor;
use crate::proxy;
use crate::registry::exposed_id;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/tags", get(list_tags))
        .route("/api/generate", post(proxy_post))
        .route("/api/chat", post(proxy_post))
}

async fn list_tags(State(state): State<AppState>) -> Response {
    let mut models: Vec<serde_json::Value> = state
        .registry
        .list_models(Flavor::Ollama)
        .into_iter()
        .map(|(server, model)| {
            let mut raw = model.raw.clone();
            let id = exposed_id(&server, &model.upstream_name);
            raw["name"] = serde_json::Value::String(id.clone());
            raw["model"] = serde_json::Value::String(id);
            raw
        })
        .collect();
    models.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Json(serde_json::json!({"models": models})).into_response()
}

async fn proxy_post(State(state): State<AppState>, uri: axum::http::Uri, body: Bytes) -> Response {
    proxy::dispatch(state, uri, body, Flavor::Ollama, error).await
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": message}))).into_response()
}
