mod common;

use local_llm_router::config::{Config, Flavor, RouterConfig, ServerConfig};
use local_llm_router::{AppState, discovery};

fn config_with(servers: Vec<ServerConfig>) -> Config {
    Config {
        router: RouterConfig::default(),
        server: servers,
    }
}

#[tokio::test]
async fn fetches_openai_models() {
    let (base, _cap) = common::mock_openai(vec!["gpt-oss-20b", "qwen3-32b"]).await;
    let http = reqwest::Client::new();
    let server = ServerConfig {
        name: "studio".into(),
        url: base,
        flavor: Flavor::OpenAi,
        models: None,
    };
    let models = discovery::fetch_models(&http, &server).await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].upstream_name, "gpt-oss-20b");
    assert_eq!(models[0].raw["object"], "model");
}

#[tokio::test]
async fn fetches_ollama_models() {
    let (base, _cap) = common::mock_ollama(vec!["qwen3:8b"]).await;
    let http = reqwest::Client::new();
    let server = ServerConfig {
        name: "rig".into(),
        url: base,
        flavor: Flavor::Ollama,
        models: None,
    };
    let models = discovery::fetch_models(&http, &server).await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].upstream_name, "qwen3:8b");
    assert_eq!(models[0].raw["details"]["family"], "qwen");
}

#[tokio::test]
async fn applies_allowlist() {
    let (base, _cap) = common::mock_openai(vec!["a", "b"]).await;
    let http = reqwest::Client::new();
    let server = ServerConfig {
        name: "studio".into(),
        url: base,
        flavor: Flavor::OpenAi,
        models: Some(vec!["b".into()]),
    };
    let models = discovery::fetch_models(&http, &server).await.unwrap();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].upstream_name, "b");
}

#[tokio::test]
async fn refresh_marks_unreachable_server_without_failing() {
    let (base, _cap) = common::mock_openai(vec!["m"]).await;
    let servers = vec![
        ServerConfig {
            name: "up".into(),
            url: base,
            flavor: Flavor::OpenAi,
            models: None,
        },
        ServerConfig {
            name: "down".into(),
            url: "http://127.0.0.1:1".into(), // nothing listening
            flavor: Flavor::OpenAi,
            models: None,
        },
    ];
    let state = AppState::new(&config_with(servers), vec![]).unwrap();
    let results = discovery::refresh_all(&state).await;
    assert_eq!(results.len(), 2);
    let up = results.iter().find(|(n, _)| n == "up").unwrap();
    let down = results.iter().find(|(n, _)| n == "down").unwrap();
    assert!(up.1);
    assert!(!down.1);
    // reachable server's models are listed; unreachable one contributes none
    let listed = state.registry.list_models(Flavor::OpenAi);
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].0, "up");
}
