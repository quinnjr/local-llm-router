mod common;

use local_llm_router::config::{Config, Flavor, RouterConfig, ServerConfig};
use local_llm_router::{AppState, api_admin, discovery};

async fn test_app(state_file: std::path::PathBuf) -> (String, AppState, common::Capture) {
    let (base, cap) = common::mock_openai(vec!["gpt-oss-20b"]).await;
    let config = Config {
        router: RouterConfig {
            state_file,
            ..RouterConfig::default()
        },
        server: vec![ServerConfig {
            name: "studio".into(),
            url: base,
            flavor: Flavor::OpenAi,
            models: None,
        }],
    };
    let state = AppState::new(&config, vec![]).unwrap();
    discovery::refresh_all(&state).await;
    let url = common::spawn(api_admin::router().with_state(state.clone())).await;
    (url, state, cap)
}

fn temp_state(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("llr-admin-{tag}-{}.toml", std::process::id()))
}

#[tokio::test]
async fn health_is_ok() {
    let (url, _s, _c) = test_app(temp_state("health")).await;
    let resp = reqwest::get(format!("{url}/health")).await.unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(resp.text().await.unwrap(), "ok");
}

#[tokio::test]
async fn lists_servers_with_models_and_origin() {
    let (url, _s, _c) = test_app(temp_state("list")).await;
    let body: serde_json::Value = reqwest::get(format!("{url}/admin/servers"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let servers = body["servers"].as_array().unwrap();
    assert_eq!(servers[0]["name"], "studio");
    assert_eq!(servers[0]["flavor"], "openai");
    assert_eq!(servers[0]["origin"], "config");
    assert_eq!(servers[0]["reachable"], true);
    assert_eq!(servers[0]["models"][0], "studio/gpt-oss-20b");
}

#[tokio::test]
async fn add_server_persists_and_discovers() {
    let state_file = temp_state("add");
    let (url, _s, _c) = test_app(state_file.clone()).await;
    let (new_base, _new_cap) = common::mock_ollama(vec!["qwen3:8b"]).await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/admin/servers"))
        .json(&serde_json::json!({"name": "rig", "url": new_base, "flavor": "ollama"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 201);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["origin"], "state");
    assert_eq!(body["models"][0], "rig/qwen3:8b");
    // persisted to the state file
    let text = std::fs::read_to_string(&state_file).unwrap();
    assert!(text.contains("rig"));
    std::fs::remove_file(&state_file).ok();
}

#[tokio::test]
async fn add_duplicate_name_is_409() {
    let (url, _s, _c) = test_app(temp_state("dup")).await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/admin/servers"))
        .json(&serde_json::json!({"name": "studio", "url": "http://x:1", "flavor": "openai"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
}

#[tokio::test]
async fn add_slash_name_is_409() {
    let (url, _s, _c) = test_app(temp_state("slash")).await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/admin/servers"))
        .json(&serde_json::json!({"name": "a/b", "url": "http://x:1", "flavor": "openai"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
}

#[tokio::test]
async fn add_bad_url_is_400() {
    let (url, _s, _c) = test_app(temp_state("badurl")).await;
    let resp = reqwest::Client::new()
        .post(format!("{url}/admin/servers"))
        .json(&serde_json::json!({"name": "x", "url": "not a url", "flavor": "openai"}))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn delete_config_server_is_409_state_server_is_204() {
    let state_file = temp_state("del");
    let (url, _s, _c) = test_app(state_file.clone()).await;
    let client = reqwest::Client::new();
    // config-defined: rejected
    let resp = client
        .delete(format!("{url}/admin/servers/studio"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 409);
    // add a state server, then delete it
    client
        .post(format!("{url}/admin/servers"))
        .json(&serde_json::json!({"name": "tmp", "url": "http://127.0.0.1:1", "flavor": "openai"}))
        .send()
        .await
        .unwrap();
    let resp = client
        .delete(format!("{url}/admin/servers/tmp"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 204);
    // state file no longer contains it
    let text = std::fs::read_to_string(&state_file).unwrap_or_default();
    assert!(!text.contains("tmp"));
    // deleting again is 404
    let resp = client
        .delete(format!("{url}/admin/servers/tmp"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    std::fs::remove_file(&state_file).ok();
}

#[tokio::test]
async fn refresh_reports_reachability() {
    let (url, _s, _c) = test_app(temp_state("refresh")).await;
    let body: serde_json::Value = reqwest::Client::new()
        .post(format!("{url}/admin/refresh"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results[0]["name"], "studio");
    assert_eq!(results[0]["reachable"], true);
}
