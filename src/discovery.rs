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
    let mut results = vec![];
    for entry in servers {
        let name = entry.config.name.clone();
        let reachable = refresh_discovered(state, &entry.config).await;
        results.push((name, reachable));
    }
    results
}

pub async fn refresh_one(state: &AppState, name: &str) {
    if let Some(entry) = state
        .registry
        .list_servers()
        .into_iter()
        .find(|e| e.config.name == name)
    {
        refresh_discovered(state, &entry.config).await;
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
