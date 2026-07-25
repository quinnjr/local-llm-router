use crate::AppState;
use crate::config::Flavor;
use crate::proxy::{self, ForwardError};
use crate::registry::{ResolveError, exposed_id};
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(proxy_post))
        .route("/v1/completions", post(proxy_post))
        .route("/v1/embeddings", post(proxy_post))
}

async fn list_models(State(state): State<AppState>) -> Response {
    let mut data: Vec<serde_json::Value> = state
        .registry
        .list_models(Flavor::OpenAi)
        .into_iter()
        .map(|(server, model)| {
            let mut raw = model.raw.clone();
            raw["id"] = serde_json::Value::String(exposed_id(&server, &model.upstream_name));
            raw["owned_by"] = serde_json::Value::String(server);
            raw
        })
        .collect();
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}

async fn proxy_post(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    body: Bytes,
) -> Response {
    let path = uri.path().to_string();
    let Some(model) = proxy::extract_model(&body) else {
        return error(StatusCode::BAD_REQUEST, "request body has no \"model\" field");
    };
    let resolved = match state.registry.resolve(Flavor::OpenAi, &model) {
        Ok(r) => r,
        Err(ResolveError::Unknown { known, .. }) => {
            return error(
                StatusCode::NOT_FOUND,
                &format!("unknown model '{model}'. known models: {}", known.join(", ")),
            );
        }
        Err(ResolveError::Ambiguous { candidates, .. }) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!(
                    "ambiguous model '{model}'. qualified candidates: {}",
                    candidates.join(", ")
                ),
            );
        }
    };
    match proxy::forward(&state, &resolved, &path, body).await {
        Ok(resp) => resp,
        Err(e) => {
            let status = match e {
                ForwardError::MissingModel | ForwardError::BadJson => StatusCode::BAD_REQUEST,
                ForwardError::Upstream(_) => StatusCode::BAD_GATEWAY,
            };
            error(status, &e.to_string())
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": "invalid_request_error", "code": null}
        })),
    )
        .into_response()
}
