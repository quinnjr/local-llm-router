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
