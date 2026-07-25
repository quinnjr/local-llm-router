// Per-target helper usage varies; each test binary compiles its own copy.
#![allow(dead_code)]

use axum::{Json, Router, routing::get, routing::post};
use std::sync::{Arc, Mutex};

/// Spawn any router on an ephemeral port; returns the base URL.
pub async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[derive(Clone, Default)]
pub struct Capture {
    pub last_model: Arc<Mutex<Option<String>>>,
}

/// Mock llama.cpp-style server: GET /v1/models + POST /v1/chat/completions.
/// The POST captures the `model` field and replies with a two-chunk SSE body.
pub async fn mock_openai(models: Vec<&str>) -> (String, Capture) {
    let capture = Capture::default();
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| serde_json::json!({"id": m, "object": "model", "created": 0, "owned_by": "mock"}))
        .collect();
    let cap = capture.clone();
    let router = Router::new()
        .route(
            "/v1/models",
            get(move || {
                let data = data.clone();
                async move { Json(serde_json::json!({"object": "list", "data": data})) }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                async move {
                    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    *cap.last_model.lock().unwrap() =
                        v["model"].as_str().map(|s| s.to_string());
                    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
                    (
                        [("content-type", "text/event-stream")],
                        sse.to_string(),
                    )
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture)
}

/// Mock Ollama server: GET /api/tags + POST /api/chat (NDJSON reply).
pub async fn mock_ollama(models: Vec<&str>) -> (String, Capture) {
    let capture = Capture::default();
    let entries: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "name": m, "model": m, "modified_at": "2026-07-25T00:00:00Z",
                "size": 1000, "digest": "abc", "details": {"family": "qwen"}
            })
        })
        .collect();
    let cap = capture.clone();
    let router = Router::new()
        .route(
            "/api/tags",
            get(move || {
                let entries = entries.clone();
                async move { Json(serde_json::json!({"models": entries})) }
            }),
        )
        .route(
            "/api/chat",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                async move {
                    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    *cap.last_model.lock().unwrap() =
                        v["model"].as_str().map(|s| s.to_string());
                    let ndjson = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n{\"done\":true}\n";
                    ([("content-type", "application/x-ndjson")], ndjson.to_string())
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture)
}
