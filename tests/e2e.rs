mod common;

use local_llm_router::config::{Config, Flavor, RouterConfig, ServerConfig};
use local_llm_router::{AppState, build_router, discovery};

#[tokio::test]
async fn full_stack_routes_both_surfaces() {
    let (openai_base, openai_cap) = common::mock_openai(vec!["gpt-oss-20b"]).await;
    let (ollama_base, ollama_cap) = common::mock_ollama(vec!["qwen3:8b"]).await;
    let state_file =
        std::env::temp_dir().join(format!("llr-e2e-{}.toml", std::process::id()));
    let config = Config {
        router: RouterConfig {
            state_file: state_file.clone(),
            ..RouterConfig::default()
        },
        server: vec![
            ServerConfig {
                name: "studio".into(),
                url: openai_base,
                flavor: Flavor::OpenAi,
                models: None,
            },
            ServerConfig {
                name: "rig".into(),
                url: ollama_base,
                flavor: Flavor::Ollama,
                models: None,
            },
        ],
    };
    let state = AppState::new(&config, vec![]).unwrap();
    discovery::refresh_all(&state).await;
    let url = common::spawn(build_router(state)).await;
    let client = reqwest::Client::new();

    // OpenAI surface lists only openai models
    let body: serde_json::Value = client
        .get(format!("{url}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["studio/gpt-oss-20b"]);

    // Ollama surface lists only ollama models
    let body: serde_json::Value = client
        .get(format!("{url}/api/tags"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let names: Vec<&str> = body["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, vec!["rig/qwen3:8b"]);

    // OpenAI chat completion routes and rewrites
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "studio/gpt-oss-20b", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        openai_cap.last_model.lock().unwrap().as_deref(),
        Some("gpt-oss-20b")
    );

    // Ollama chat routes and rewrites
    let resp = client
        .post(format!("{url}/api/chat"))
        .json(&serde_json::json!({"model": "rig/qwen3:8b", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        ollama_cap.last_model.lock().unwrap().as_deref(),
        Some("qwen3:8b")
    );

    // Cross-surface requests do not leak: ollama model is unknown on /v1
    let resp = client
        .post(format!("{url}/v1/chat/completions"))
        .json(&serde_json::json!({"model": "rig/qwen3:8b", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);

    // Health works on the merged router
    let resp = client.get(format!("{url}/health")).send().await.unwrap();
    assert_eq!(resp.status(), 200);

    std::fs::remove_file(&state_file).ok();
}
