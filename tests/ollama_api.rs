mod common;

use local_llm_router::config::{Config, Flavor, RouterConfig, ServerConfig};
use local_llm_router::{AppState, api_ollama, discovery};

async fn test_app() -> (String, common::Capture) {
    let (base, cap) = common::mock_ollama(vec!["qwen3:8b"]).await;
    let config = Config {
        router: RouterConfig::default(),
        server: vec![ServerConfig {
            name: "rig".into(),
            url: base,
            flavor: Flavor::Ollama,
            models: None,
        }],
    };
    let state = AppState::new(&config, vec![]).unwrap();
    discovery::refresh_all(&state).await;
    let url = common::spawn(api_ollama::router().with_state(state)).await;
    (url, cap)
}

#[tokio::test]
async fn lists_namespaced_tags() {
    let (url, _cap) = test_app().await;
    let body: serde_json::Value = reqwest::get(format!("{url}/api/tags"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let models = body["models"].as_array().unwrap();
    assert_eq!(models[0]["name"], "rig/qwen3:8b");
    assert_eq!(models[0]["model"], "rig/qwen3:8b");
    // rich upstream fields survive
    assert_eq!(models[0]["details"]["family"], "qwen");
}

#[tokio::test]
async fn routes_chat_and_rewrites_model() {
    let (url, cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/api/chat"))
        .json(&serde_json::json!({"model": "rig/qwen3:8b", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("\"done\":true"));
    assert_eq!(cap.last_model.lock().unwrap().as_deref(), Some("qwen3:8b"));
}

#[tokio::test]
async fn unknown_model_is_404_with_ollama_error_shape() {
    let (url, _cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/api/generate"))
        .json(&serde_json::json!({"model": "nope", "prompt": "hi"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("rig/qwen3:8b"));
}

#[tokio::test]
async fn missing_model_field_is_400() {
    let (url, _cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/api/chat"))
        .json(&serde_json::json!({"messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().unwrap().contains("model"));
}

#[tokio::test]
async fn ambiguous_bare_model_is_400_with_candidates() {
    let (base1, _c1) = common::mock_ollama(vec!["shared"]).await;
    let (base2, _c2) = common::mock_ollama(vec!["shared"]).await;
    let url = common::spawn_app(
        vec![
            ServerConfig {
                name: "r1".into(),
                url: base1,
                flavor: Flavor::Ollama,
                models: None,
            },
            ServerConfig {
                name: "r2".into(),
                url: base2,
                flavor: Flavor::Ollama,
                models: None,
            },
        ],
        api_ollama::router(),
    )
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/api/chat"))
        .json(&serde_json::json!({"model": "shared", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"].as_str().unwrap();
    assert!(msg.contains("r1/shared"), "message missing r1/shared: {msg}");
    assert!(msg.contains("r2/shared"), "message missing r2/shared: {msg}");
}

#[tokio::test]
async fn unreachable_server_is_502() {
    let url = common::spawn_app(
        vec![ServerConfig {
            name: "down".into(),
            url: "http://127.0.0.1:1".into(),
            flavor: Flavor::Ollama,
            models: None,
        }],
        api_ollama::router(),
    )
    .await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/api/chat"))
        .json(&serde_json::json!({"model": "down/anything", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 502);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"].as_str().unwrap();
    assert!(
        msg.contains("upstream request failed"),
        "unexpected message: {msg}"
    );
}

#[tokio::test]
async fn malformed_json_is_400() {
    let (url, _cap) = test_app().await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/api/chat"))
        .header("content-type", "application/json")
        .body("not json")
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    let msg = body["error"].as_str().unwrap();
    assert!(msg.contains("not valid JSON"), "unexpected message: {msg}");
}
