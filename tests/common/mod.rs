// Per-target helper usage varies; each test binary compiles its own copy.
#![allow(dead_code)]

use axum::{Json, Router, routing::get, routing::post};
use futures_util::StreamExt as _;
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

/// Build an app over the given servers, run discovery, spawn it.
pub async fn spawn_app(
    servers: Vec<local_llm_router::config::ServerConfig>,
    router: axum::Router<local_llm_router::AppState>,
) -> String {
    let config = local_llm_router::config::Config {
        router: local_llm_router::config::RouterConfig::default(),
        server: servers,
    };
    let state = local_llm_router::AppState::new(&config, vec![]).unwrap();
    local_llm_router::discovery::refresh_all(&state).await;
    spawn(router.with_state(state)).await
}

#[derive(Clone, Default)]
pub struct Capture {
    pub last_model: Arc<Mutex<Option<String>>>,
}

/// First SSE chunk every mock chat stream starts with.
const SSE_HI: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";

/// Build the `data` array for a mock GET /v1/models reply.
fn openai_models_data(models: &[&str]) -> Vec<serde_json::Value> {
    models
        .iter()
        .map(|m| serde_json::json!({"id": m, "object": "model", "created": 0, "owned_by": "mock"}))
        .collect()
}

/// GET /v1/models handler serving the given model entries.
fn openai_models_route(data: Vec<serde_json::Value>) -> axum::routing::MethodRouter {
    get(move || {
        let data = data.clone();
        async move { Json(serde_json::json!({"object": "list", "data": data})) }
    })
}

/// Parse the request body and record its `model` field.
fn capture_model(cap: &Capture, body: &axum::body::Bytes) {
    let v: serde_json::Value = serde_json::from_slice(body).unwrap();
    *cap.last_model.lock().unwrap() = v["model"].as_str().map(|s| s.to_string());
}

/// Mock llama.cpp-style server: GET /v1/models + POST /v1/chat/completions.
/// The POST captures the `model` field and replies with a two-chunk SSE body.
pub async fn mock_openai(models: Vec<&str>) -> (String, Capture) {
    let capture = Capture::default();
    let data = openai_models_data(&models);
    let cap_chat = capture.clone();
    let cap_completions = capture.clone();
    let cap_embeddings = capture.clone();
    let router = Router::new()
        .route("/v1/models", openai_models_route(data))
        .route(
            "/v1/chat/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap_chat.clone();
                async move {
                    capture_model(&cap, &body);
                    let sse = format!("{SSE_HI}data: [DONE]\n\n");
                    (
                        [("content-type", "text/event-stream")],
                        sse.to_string(),
                    )
                }
            }),
        )
        .route(
            "/v1/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap_completions.clone();
                async move {
                    capture_model(&cap, &body);
                    Json(serde_json::json!({
                        "id": "cmpl-1", "object": "text_completion",
                        "choices": [{"text": "hi", "index": 0, "finish_reason": "stop"}]
                    }))
                }
            }),
        )
        .route(
            "/v1/embeddings",
            post(move |body: axum::body::Bytes| {
                let cap = cap_embeddings.clone();
                async move {
                    capture_model(&cap, &body);
                    Json(serde_json::json!({
                        "object": "list",
                        "data": [{"object": "embedding", "embedding": [0.0], "index": 0}]
                    }))
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
                    capture_model(&cap, &body);
                    let ndjson = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n{\"done\":true}\n";
                    ([("content-type", "application/x-ndjson")], ndjson.to_string())
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture)
}

/// Like `mock_openai`, but the chat stream yields its first SSE chunk
/// immediately, then pends until the returned oneshot is signaled, then
/// yields `data: [DONE]`. Lets tests prove the proxy streams incrementally.
pub async fn mock_openai_trickle(
    models: Vec<&str>,
) -> (String, Capture, tokio::sync::oneshot::Sender<()>) {
    let capture = Capture::default();
    let data = openai_models_data(&models);
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let rx = Arc::new(Mutex::new(Some(rx)));
    let cap = capture.clone();
    let router = Router::new()
        .route("/v1/models", openai_models_route(data))
        .route(
            "/v1/chat/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                let rx = rx.clone();
                async move {
                    capture_model(&cap, &body);
                    let rx = rx.lock().unwrap().take();
                    let first = futures_util::stream::iter([
                        Ok::<&'static str, std::convert::Infallible>(SSE_HI),
                    ]);
                    let rest = futures_util::stream::once(async move {
                        if let Some(rx) = rx {
                            rx.await.ok();
                        }
                        Ok::<&'static str, std::convert::Infallible>("data: [DONE]\n\n")
                    });
                    let stream = first.chain(rest);
                    (
                        [("content-type", "text/event-stream")],
                        axum::body::Body::from_stream(stream),
                    )
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture, tx)
}

/// Like `mock_openai`, but the chat stream yields one chunk then pends
/// forever. The stream holds a guard whose Drop flips the returned flag, so
/// a test can observe the proxy tearing the connection down on disconnect.
pub async fn mock_openai_hanging(
    models: Vec<&str>,
) -> (String, Capture, Arc<std::sync::atomic::AtomicBool>) {
    let capture = Capture::default();
    let data = openai_models_data(&models);
    let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let cap = capture.clone();
    let flag = dropped.clone();
    let router = Router::new()
        .route("/v1/models", openai_models_route(data))
        .route(
            "/v1/chat/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                let flag = flag.clone();
                async move {
                    capture_model(&cap, &body);
                    struct DropFlag(Arc<std::sync::atomic::AtomicBool>);
                    impl Drop for DropFlag {
                        fn drop(&mut self) {
                            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                    let guard = DropFlag(flag);
                    let first = futures_util::stream::iter([
                        Ok::<&'static str, std::convert::Infallible>(SSE_HI),
                    ]);
                    let hang = futures_util::stream::pending::<
                        Result<&'static str, std::convert::Infallible>,
                    >();
                    let stream = first.chain(hang).map(move |item| {
                        let _ = &guard;
                        item
                    });
                    (
                        [("content-type", "text/event-stream")],
                        axum::body::Body::from_stream(stream),
                    )
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture, dropped)
}

/// Mock whose chat endpoint yields one SSE chunk, then a stream error —
/// exercises the proxy's "truncated, logged" mid-stream failure path.
pub async fn mock_openai_failing_stream(models: Vec<&str>) -> (String, Capture) {
    let capture = Capture::default();
    let data = openai_models_data(&models);
    let cap = capture.clone();
    let router = Router::new()
        .route("/v1/models", openai_models_route(data))
        .route(
            "/v1/chat/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                async move {
                    capture_model(&cap, &body);
                    let first = futures_util::stream::iter(vec![
                        Ok::<axum::body::Bytes, std::io::Error>(
                            axum::body::Bytes::from_static(SSE_HI.as_bytes()),
                        ),
                    ]);
                    // Yield so hyper flushes the first chunk before the error.
                    let rest = futures_util::stream::once(async {
                        tokio::task::yield_now().await;
                        Err::<axum::body::Bytes, std::io::Error>(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "boom",
                        ))
                    });
                    let stream = first.chain(rest);
                    (
                        [("content-type", "text/event-stream")],
                        axum::body::Body::from_stream(stream),
                    )
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture)
}
