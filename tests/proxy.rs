mod common;

use axum::body::Bytes;
use local_llm_router::config::{Config, Flavor, RouterConfig, ServerConfig};
use local_llm_router::registry::Resolved;
use local_llm_router::{AppState, proxy};

fn app_state() -> AppState {
    let config = Config {
        router: RouterConfig::default(),
        server: vec![],
    };
    AppState::new(&config, vec![]).unwrap()
}

#[tokio::test]
async fn rewrites_model_field_and_streams_response() {
    let (base, cap) = common::mock_openai(vec!["gpt-oss-20b"]).await;
    let state = app_state();
    let resolved = Resolved {
        server_url: base,
        upstream_model: "gpt-oss-20b".into(),
    };
    let body = Bytes::from(r#"{"model":"studio/gpt-oss-20b","messages":[],"stream":true}"#);
    let resp = proxy::forward(&state, &resolved, "/v1/chat/completions", body)
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()["content-type"].to_str().unwrap(),
        "text/event-stream"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.contains("[DONE]"));
    // upstream saw the bare upstream model name, not the namespaced one
    assert_eq!(
        cap.last_model.lock().unwrap().as_deref(),
        Some("gpt-oss-20b")
    );
}

#[tokio::test]
async fn missing_model_field_is_rejected() {
    let state = app_state();
    let resolved = Resolved {
        server_url: "http://unused".into(),
        upstream_model: "m".into(),
    };
    let body = Bytes::from(r#"{"messages":[]}"#);
    let err = proxy::forward(&state, &resolved, "/v1/chat/completions", body)
        .await
        .unwrap_err();
    assert!(matches!(err, proxy::ForwardError::MissingModel));
}

#[tokio::test]
async fn invalid_json_is_rejected() {
    let state = app_state();
    let resolved = Resolved {
        server_url: "http://unused".into(),
        upstream_model: "m".into(),
    };
    let body = Bytes::from("not json");
    let err = proxy::forward(&state, &resolved, "/v1/chat/completions", body)
        .await
        .unwrap_err();
    assert!(matches!(err, proxy::ForwardError::BadJson));
}

#[tokio::test]
async fn unreachable_upstream_maps_to_upstream_error() {
    let state = app_state();
    let resolved = Resolved {
        server_url: "http://127.0.0.1:1".into(),
        upstream_model: "m".into(),
    };
    let body = Bytes::from(r#"{"model":"x/m"}"#);
    let err = proxy::forward(&state, &resolved, "/v1/chat/completions", body)
        .await
        .unwrap_err();
    assert!(matches!(err, proxy::ForwardError::Upstream(_)));
}

#[test]
fn extract_model_reads_top_level_string() {
    let body = Bytes::from(r#"{"model":"a/b","messages":[]}"#);
    assert_eq!(proxy::extract_model(&body).as_deref(), Some("a/b"));
    let body = Bytes::from(r#"{"messages":[]}"#);
    assert_eq!(proxy::extract_model(&body), None);
}
