use crate::AppState;
use crate::registry::Resolved;
use axum::body::Body;
use axum::response::Response;
use futures_util::TryStreamExt;

#[derive(Debug)]
pub enum ForwardError {
    MissingModel,
    Upstream(reqwest::Error),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::MissingModel => write!(f, "request body has no \"model\" field"),
            ForwardError::Upstream(e) => write!(f, "upstream request failed: {e}"),
        }
    }
}

impl std::error::Error for ForwardError {}

pub fn extract_model(json: &serde_json::Value) -> Option<&str> {
    json["model"].as_str()
}

/// Shared parse → resolve → forward flow for both API surfaces.
/// `error` renders the surface-native error envelope.
pub async fn dispatch(
    state: AppState,
    uri: axum::http::Uri,
    body: axum::body::Bytes,
    flavor: crate::config::Flavor,
    error: fn(axum::http::StatusCode, &str) -> axum::response::Response,
) -> axum::response::Response {
    let path = uri.path().to_string();
    let json: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(j) => j,
        Err(_) => {
            return error(
                axum::http::StatusCode::BAD_REQUEST,
                "request body is not valid JSON",
            );
        }
    };
    let Some(model) = extract_model(&json) else {
        return error(
            axum::http::StatusCode::BAD_REQUEST,
            "request body has no \"model\" field",
        );
    };
    let resolved = match state.registry.resolve(flavor, model) {
        Ok(r) => r,
        Err(crate::registry::ResolveError::Unknown { known, .. }) => {
            return error(
                axum::http::StatusCode::NOT_FOUND,
                &format!("unknown model '{model}'. known models: {}", known.join(", ")),
            );
        }
        Err(crate::registry::ResolveError::Ambiguous { candidates, .. }) => {
            return error(
                axum::http::StatusCode::BAD_REQUEST,
                &format!(
                    "ambiguous model '{model}'. qualified candidates: {}",
                    candidates.join(", ")
                ),
            );
        }
    };
    match forward(&state, &resolved, &path, json).await {
        Ok(resp) => resp,
        Err(e) => {
            let status = match e {
                ForwardError::MissingModel => axum::http::StatusCode::BAD_REQUEST,
                ForwardError::Upstream(_) => axum::http::StatusCode::BAD_GATEWAY,
            };
            error(status, &e.to_string())
        }
    }
}

pub async fn forward(
    state: &AppState,
    resolved: &Resolved,
    path: &str,
    mut json: serde_json::Value,
) -> Result<Response, ForwardError> {
    if json["model"].as_str().is_none() {
        return Err(ForwardError::MissingModel);
    }
    json["model"] = serde_json::Value::String(resolved.upstream_model.clone());

    let url = format!("{}{}", resolved.server_url.trim_end_matches('/'), path);
    let upstream = state
        .http
        .post(&url)
        .json(&json)
        .send()
        .await
        .map_err(ForwardError::Upstream)?;

    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    // Byte-stream pass-through. When the client disconnects, axum drops this
    // body, which drops the reqwest stream and cancels the upstream request.
    let server = resolved.server_url.clone();
    let path = path.to_string();
    let body = Body::from_stream(upstream.bytes_stream().inspect_err(move |e| {
        tracing::warn!(server = %server, path = %path, error = %e, "upstream stream failed mid-response");
    }));
    Ok(Response::builder()
        .status(status.as_u16())
        .header("content-type", content_type)
        .body(body)
        .expect("response builder failed"))
}
