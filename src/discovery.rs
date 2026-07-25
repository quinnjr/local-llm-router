use crate::AppState;
use crate::config::{Flavor, ServerConfig};
use crate::registry::DiscoveredModel;

pub async fn fetch_models(
    http: &reqwest::Client,
    server: &ServerConfig,
) -> Result<Vec<DiscoveredModel>, reqwest::Error> {
    let base = server.url.trim_end_matches('/');
    let url = match server.flavor {
        Flavor::OpenAi => format!("{base}/v1/models"),
        Flavor::Ollama => format!("{base}/api/tags"),
    };
    let body: serde_json::Value = http.get(&url).send().await?.json().await?;
    let items: &[serde_json::Value] = match server.flavor {
        Flavor::OpenAi => body["data"].as_array().map(Vec::as_slice).unwrap_or(&[]),
        Flavor::Ollama => body["models"].as_array().map(Vec::as_slice).unwrap_or(&[]),
    };
    let name_key = match server.flavor {
        Flavor::OpenAi => "id",
        Flavor::Ollama => "name",
    };
    let mut out = vec![];
    for item in items {
        let Some(name) = item[name_key].as_str() else {
            continue;
        };
        if let Some(allow) = &server.models {
            if !allow.iter().any(|a| a == name) {
                continue;
            }
        }
        out.push(DiscoveredModel {
            upstream_name: name.to_string(),
            raw: item.clone(),
        });
    }
    Ok(out)
}

pub async fn refresh_all(state: &AppState) -> Vec<(String, bool)> {
    let servers = state.registry.list_servers();
    let mut set = tokio::task::JoinSet::new();
    for entry in servers {
        let state = state.clone();
        set.spawn(async move {
            let name = entry.config.name.clone();
            let reachable = refresh_discovered(&state, &entry.config).await;
            (name, reachable)
        });
    }
    let mut results = vec![];
    while let Some(res) = set.join_next().await {
        match res {
            Ok(pair) => results.push(pair),
            Err(e) => tracing::error!(error = %e, "discovery task failed; refresh results incomplete"),
        }
    }
    results.sort_by(|a, b| a.0.cmp(&b.0)); // deterministic output (tests rely on it)
    assert_unique_exposed_ids(state);
    results
}

pub async fn refresh_one(state: &AppState, name: &str) {
    if let Some(entry) = state.registry.get_server(name) {
        refresh_discovered(state, &entry.config).await;
    }
    assert_unique_exposed_ids(state);
}

/// Defensive uniqueness check per the spec: duplicate exposed model ids are
/// impossible by construction; log loudly (and panic in debug) if one appears.
fn assert_unique_exposed_ids(state: &AppState) {
    let mut ids: Vec<String> = state
        .registry
        .list_models(crate::config::Flavor::OpenAi)
        .into_iter()
        .chain(state.registry.list_models(crate::config::Flavor::Ollama))
        .map(|(server, m)| crate::registry::exposed_id(&server, &m.upstream_name))
        .collect();
    ids.sort();
    for w in ids.windows(2) {
        if w[0] == w[1] {
            tracing::error!(id = %w[0], "duplicate exposed model id after discovery refresh");
            debug_assert!(false, "duplicate exposed model id: {}", w[0]);
        }
    }
}

/// Returns reachability; updates the registry as a side effect.
async fn refresh_discovered(state: &AppState, server: &ServerConfig) -> bool {
    match fetch_models(&state.http, server).await {
        Ok(models) => {
            state.registry.set_models(&server.name, Some(models));
            true
        }
        Err(e) => {
            tracing::warn!(server = %server.name, error = %e, "model discovery failed");
            state.registry.set_models(&server.name, None);
            false
        }
    }
}
