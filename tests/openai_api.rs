mod common;

use local_llm_router::config::{Config, Flavor, RouterConfig, ServerConfig};
use local_llm_router::{AppState, api_openai, discovery};

async fn test_app() -> (String, common::Capture) {
    let (base, cap) = common::mock_openai(vec!["gpt-oss-20b"]).await;
    let config = Config {
        router: RouterConfig::default(),
        server: vec![ServerConfig {
            name: "studio".into(),
            url: base,
            flavor: Flavor::OpenAi,
            models: None,
        }],
    };
    let state = AppState::new(&config, vec![]).unwrap();
    discovery::refresh_all(&state).await;
    let url = common::spawn(api_openai::router().with_state(state)).await;
    (url, cap)
}

#[tokio::test]
async fn lists_namespaced_models() {
    let (url, _cap) = test_app().await;
    let body: serde_json::Value = reqwest::get(format!("{url}/v1/models"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data[0]["id"], "studio/gpt-oss-20b");
    assert_eq!(data[0]["owned_by"], "studio");
}

#[tokio::test]
async fn routes_chat_completions_and_rewrites_model() {
    let (url, cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "studio/gpt-oss-20b", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("[DONE]"));
    assert_eq!(
        cap.last_model.lock().unwrap().as_deref(),
        Some("gpt-oss-20b")
    );
}

#[tokio::test]
async fn unknown_model_is_404_with_openai_error_shape() {
    let (url, _cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "nope", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("studio/gpt-oss-20b"));
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn bare_model_resolves() {
    let (url, cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "gpt-oss-20b", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        cap.last_model.lock().unwrap().as_deref(),
        Some("gpt-oss-20b")
    );
}

#[tokio::test]
async fn missing_model_field_is_400() {
    let (url, _cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().is_some());
}

#[tokio::test]
async fn streams_first_chunk_before_upstream_completes() {
    let (base, _cap, gate) = common::mock_openai_trickle(vec!["m"]).await;
    let url = common::spawn_app(
        vec![ServerConfig {
            name: "s".into(),
            url: base,
            flavor: Flavor::OpenAi,
            models: None,
        }],
        api_openai::router(),
    )
    .await;
    let client = reqwest::Client::new();
    let mut resp = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "s/m", "messages": []}))
            .send(),
    )
    .await
    .expect("proxy buffered before sending headers")
    .unwrap();
    assert_eq!(resp.status(), 200);
    // The first chunk must arrive while the upstream body is still gated;
    // a buffering proxy would pend here until the gate is released.
    let first = tokio::time::timeout(std::time::Duration::from_secs(2), resp.chunk())
        .await
        .expect("proxy buffered the upstream body instead of streaming")
        .unwrap()
        .expect("stream ended before the first chunk");
    assert!(!first.is_empty());
    gate.send(()).unwrap();
    let mut rest = Vec::new();
    while let Some(chunk) = resp.chunk().await.unwrap() {
        rest.extend_from_slice(&chunk);
    }
    assert!(String::from_utf8_lossy(&rest).contains("[DONE]"));
}

#[tokio::test]
async fn client_disconnect_cancels_upstream() {
    let (base, _cap, dropped) = common::mock_openai_hanging(vec!["m"]).await;
    let url = common::spawn_app(
        vec![ServerConfig {
            name: "s".into(),
            url: base,
            flavor: Flavor::OpenAi,
            models: None,
        }],
        api_openai::router(),
    )
    .await;
    let client = reqwest::Client::new();
    let mut resp = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        client
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "s/m", "messages": []}))
            .send(),
    )
    .await
    .expect("proxy buffered before sending headers")
    .unwrap();
    assert_eq!(resp.status(), 200);
    let first = resp.chunk().await.unwrap();
    assert!(first.is_some());
    // Negative control: while the client is still connected, the upstream
    // connection must not be torn down.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert!(
        !dropped.load(std::sync::atomic::Ordering::SeqCst),
        "upstream torn down while client still connected"
    );
    drop(resp);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !dropped.load(std::sync::atomic::Ordering::SeqCst) {
        assert!(
            std::time::Instant::now() < deadline,
            "upstream connection was not torn down after client disconnect"
        );
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

#[tokio::test]
async fn mid_stream_failure_truncates_response() {
    let (base, _cap) = common::mock_openai_failing_stream(vec!["m"]).await;
    let url = common::spawn_app(
        vec![ServerConfig {
            name: "s".into(),
            url: base,
            flavor: Flavor::OpenAi,
            models: None,
        }],
        api_openai::router(),
    )
    .await;
    let mut resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "s/m", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    // Upstream dies after the first chunk: the client sees the stream cut
    // short (a terminal read error), never a [DONE] sentinel.
    let mut body = Vec::new();
    while let Ok(Some(chunk)) = resp.chunk().await {
        body.extend_from_slice(&chunk);
    }
    let body = String::from_utf8_lossy(&body);
    assert!(body.contains("hi"), "first chunk missing: {body}");
    assert!(!body.contains("[DONE]"), "stream was not truncated: {body}");
}

#[tokio::test]
async fn ambiguous_bare_model_is_400_with_candidates() {
    let (base1, _c1) = common::mock_openai(vec!["shared"]).await;
    let (base2, _c2) = common::mock_openai(vec!["shared"]).await;
    let url = common::spawn_app(
        vec![
            ServerConfig {
                name: "s1".into(),
                url: base1,
                flavor: Flavor::OpenAi,
                models: None,
            },
            ServerConfig {
                name: "s2".into(),
                url: base2,
                flavor: Flavor::OpenAi,
                models: None,
            },
        ],
        api_openai::router(),
    )
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "shared", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("s1/shared"), "message missing s1/shared: {msg}");
    assert!(msg.contains("s2/shared"), "message missing s2/shared: {msg}");
}

#[tokio::test]
async fn unreachable_server_is_502() {
    let url = common::spawn_app(
        vec![ServerConfig {
            name: "down".into(),
            url: "http://127.0.0.1:1".into(),
            flavor: Flavor::OpenAi,
            models: None,
        }],
        api_openai::router(),
    )
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "down/anything", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(
        msg.contains("upstream request failed"),
        "unexpected message: {msg}"
    );
}

#[tokio::test]
async fn malformed_json_is_400() {
    let (url, _cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"]["message"].as_str().unwrap();
    assert!(msg.contains("not valid JSON"), "unexpected message: {msg}");
}

#[tokio::test]
async fn completions_and_embeddings_route_to_upstream() {
    let (url, cap) = test_app().await;
    let client = reqwest::Client::new();
    for (path, payload) in [
        ("/v1/completions", serde_json::json!({"model": "studio/gpt-oss-20b", "prompt": "hi"})),
        ("/v1/embeddings", serde_json::json!({"model": "studio/gpt-oss-20b", "input": "hi"})),
    ] {
        let resp = client
            .post(format!("{url}{path}"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{path} did not return 200");
        assert_eq!(
            cap.last_model.lock().unwrap().as_deref(),
            Some("gpt-oss-20b"),
            "{path} did not forward the bare upstream model name"
        );
    }
}
