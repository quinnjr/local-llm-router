use crate::AppState;
use crate::registry::Resolved;
use axum::body::{Body, Bytes};
use axum::response::Response;

#[derive(Debug)]
pub enum ForwardError {
    MissingModel,
    BadJson,
    Upstream(reqwest::Error),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::MissingModel => write!(f, "request body has no \"model\" field"),
            ForwardError::BadJson => write!(f, "request body is not valid JSON"),
            ForwardError::Upstream(e) => write!(f, "upstream request failed: {e}"),
        }
    }
}

impl std::error::Error for ForwardError {}

pub fn extract_model(body: &Bytes) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v["model"].as_str().map(|s| s.to_string())
}

pub async fn forward(
    state: &AppState,
    resolved: &Resolved,
    path: &str,
    body: Bytes,
) -> Result<Response, ForwardError> {
    let mut json: serde_json::Value =
        serde_json::from_slice(&body).map_err(|_| ForwardError::BadJson)?;
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
    let body = Body::from_stream(upstream.bytes_stream());
    Ok(Response::builder()
        .status(status.as_u16())
        .header("content-type", content_type)
        .body(body)
        .expect("response builder failed"))
}
