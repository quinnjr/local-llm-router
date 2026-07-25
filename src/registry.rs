use crate::config::{Flavor, ServerConfig};
use std::collections::HashMap;
use std::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Config,
    State,
}

#[derive(Clone, Debug)]
pub struct DiscoveredModel {
    pub upstream_name: String,
    pub raw: serde_json::Value,
}

#[derive(Clone, Debug)]
pub struct ServerEntry {
    pub config: ServerConfig,
    pub origin: Origin,
    pub models: Option<Vec<DiscoveredModel>>,
}

impl ServerEntry {
    pub fn reachable(&self) -> bool {
        self.models.is_some()
    }
}

#[derive(Clone, Debug)]
pub struct Resolved {
    pub server_url: String,
    pub upstream_model: String,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ResolveError {
    Unknown {
        model: String,
        known: Vec<String>,
    },
    Ambiguous {
        model: String,
        candidates: Vec<String>,
    },
}

#[derive(Debug)]
pub enum RegistryError {
    DuplicateName(String),
    InvalidName(String),
    UnknownServer(String),
    ConfigOrigin(String),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::DuplicateName(n) => write!(f, "duplicate server name '{n}'"),
            RegistryError::InvalidName(n) => {
                write!(f, "server name '{n}' must not be empty or contain '/'")
            }
            RegistryError::UnknownServer(n) => write!(f, "unknown server '{n}'"),
            RegistryError::ConfigOrigin(n) => {
                write!(
                    f,
                    "server '{n}' is defined in config.toml; edit the file to remove it"
                )
            }
        }
    }
}

impl std::error::Error for RegistryError {}

pub fn exposed_id(server: &str, upstream: &str) -> String {
    format!("{server}/{upstream}")
}

#[derive(Debug)]
pub struct Registry {
    inner: RwLock<HashMap<String, ServerEntry>>,
}

impl Registry {
    pub fn new(servers: Vec<(ServerConfig, Origin)>) -> Result<Registry, RegistryError> {
        let reg = Registry {
            inner: RwLock::new(HashMap::new()),
        };
        for (config, origin) in servers {
            reg.upsert_server(config, origin)?;
        }
        Ok(reg)
    }

    pub fn resolve(&self, flavor: Flavor, model: &str) -> Result<Resolved, ResolveError> {
        let inner = self.inner.read().unwrap();
        if let Some((name, upstream)) = model.split_once('/') {
            let entry = inner.get(name).filter(|e| e.config.flavor == flavor);
            let entry = match entry {
                Some(e) => e,
                None => return Err(self.unknown_locked(&inner, flavor, model)),
            };
            let allowed = match &entry.models {
                Some(models) => models.iter().any(|m| m.upstream_name == upstream),
                None => match &entry.config.models {
                    // unreachable: allow pass-through so the proxy can 502
                    None => true,
                    Some(allow) => allow.iter().any(|a| a == upstream),
                },
            };
            if allowed {
                return Ok(Resolved {
                    server_url: entry.config.url.clone(),
                    upstream_model: upstream.to_string(),
                });
            }
            return Err(self.unknown_locked(&inner, flavor, model));
        }

        // bare name: must match exactly one discovered model on this flavor
        let mut matches = vec![];
        for (name, entry) in inner.iter() {
            if entry.config.flavor != flavor {
                continue;
            }
            if let Some(models) = &entry.models {
                if models.iter().any(|m| m.upstream_name == model) {
                    matches.push(exposed_id(name, model));
                }
            }
        }
        match matches.len() {
            1 => {
                let (name, _) = matches[0].split_once('/').unwrap();
                let entry = inner.get(name).unwrap();
                Ok(Resolved {
                    server_url: entry.config.url.clone(),
                    upstream_model: model.to_string(),
                })
            }
            0 => Err(self.unknown_locked(&inner, flavor, model)),
            _ => {
                matches.sort();
                Err(ResolveError::Ambiguous {
                    model: model.to_string(),
                    candidates: matches,
                })
            }
        }
    }

    fn unknown_locked(
        &self,
        inner: &HashMap<String, ServerEntry>,
        flavor: Flavor,
        model: &str,
    ) -> ResolveError {
        let mut known: Vec<String> = vec![];
        for (name, entry) in inner.iter() {
            if entry.config.flavor != flavor {
                continue;
            }
            if let Some(models) = &entry.models {
                for m in models {
                    known.push(exposed_id(name, &m.upstream_name));
                }
            }
        }
        known.sort();
        ResolveError::Unknown {
            model: model.to_string(),
            known,
        }
    }

    pub fn upsert_server(&self, config: ServerConfig, origin: Origin) -> Result<(), RegistryError> {
        if config.name.is_empty() || config.name.contains('/') {
            return Err(RegistryError::InvalidName(config.name));
        }
        let mut inner = self.inner.write().unwrap();
        if inner.contains_key(&config.name) {
            return Err(RegistryError::DuplicateName(config.name));
        }
        inner.insert(
            config.name.clone(),
            ServerEntry {
                config,
                origin,
                models: None,
            },
        );
        Ok(())
    }

    pub fn remove_server(&self, name: &str) -> Result<ServerEntry, RegistryError> {
        let mut inner = self.inner.write().unwrap();
        match inner.get(name) {
            None => Err(RegistryError::UnknownServer(name.to_string())),
            Some(e) if e.origin == Origin::Config => {
                Err(RegistryError::ConfigOrigin(name.to_string()))
            }
            Some(_) => Ok(inner.remove(name).unwrap()),
        }
    }

    pub fn set_models(&self, name: &str, models: Option<Vec<DiscoveredModel>>) {
        let mut inner = self.inner.write().unwrap();
        if let Some(entry) = inner.get_mut(name) {
            entry.models = models;
        }
    }

    pub fn list_models(&self, flavor: Flavor) -> Vec<(String, DiscoveredModel)> {
        let inner = self.inner.read().unwrap();
        let mut out = vec![];
        for (name, entry) in inner.iter() {
            if entry.config.flavor != flavor {
                continue;
            }
            if let Some(models) = &entry.models {
                for m in models {
                    out.push((name.clone(), m.clone()));
                }
            }
        }
        out.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.upstream_name.cmp(&b.1.upstream_name))
        });
        out
    }

    pub fn list_servers(&self) -> Vec<ServerEntry> {
        let inner = self.inner.read().unwrap();
        let mut out: Vec<_> = inner.values().cloned().collect();
        out.sort_by(|a, b| a.config.name.cmp(&b.config.name));
        out
    }

    pub fn state_servers(&self) -> Vec<ServerConfig> {
        let inner = self.inner.read().unwrap();
        inner
            .values()
            .filter(|e| e.origin == Origin::State)
            .map(|e| e.config.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn srv(name: &str, flavor: Flavor) -> ServerConfig {
        ServerConfig {
            name: name.into(),
            url: format!("http://{name}.lan:8080"),
            flavor,
            models: None,
        }
    }

    fn dm(name: &str) -> DiscoveredModel {
        DiscoveredModel {
            upstream_name: name.into(),
            raw: serde_json::json!({"id": name}),
        }
    }

    fn registry() -> Registry {
        let reg = Registry::new(vec![
            (srv("studio", Flavor::OpenAi), Origin::Config),
            (srv("rig", Flavor::Ollama), Origin::Config),
            (srv("down", Flavor::OpenAi), Origin::Config),
        ])
        .unwrap();
        reg.set_models("studio", Some(vec![dm("gpt-oss-20b")]));
        reg.set_models("rig", Some(vec![dm("qwen3:8b")]));
        // "down" keeps models: None (unreachable)
        reg
    }

    #[test]
    fn rejects_duplicate_names() {
        let err = Registry::new(vec![
            (srv("a", Flavor::OpenAi), Origin::Config),
            (srv("a", Flavor::Ollama), Origin::State),
        ])
        .unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateName(n) if n == "a"));
    }

    #[test]
    fn rejects_slash_in_name() {
        let err = Registry::new(vec![(
            ServerConfig {
                name: "a/b".into(),
                ..srv("x", Flavor::OpenAi)
            },
            Origin::Config,
        )])
        .unwrap_err();
        assert!(matches!(err, RegistryError::InvalidName(_)));
    }

    #[test]
    fn resolves_qualified_id() {
        let r = registry()
            .resolve(Flavor::OpenAi, "studio/gpt-oss-20b")
            .unwrap();
        assert_eq!(r.server_url, "http://studio.lan:8080");
        assert_eq!(r.upstream_model, "gpt-oss-20b");
    }

    #[test]
    fn resolves_qualified_upstream_name_containing_slash() {
        let reg = registry();
        reg.set_models("studio", Some(vec![dm("org/model-x")]));
        let r = reg.resolve(Flavor::OpenAi, "studio/org/model-x").unwrap();
        assert_eq!(r.upstream_model, "org/model-x");
    }

    #[test]
    fn flavor_mismatch_is_unknown() {
        // "rig" exists but is ollama-flavor; not visible on the openai surface
        let err = registry()
            .resolve(Flavor::OpenAi, "rig/qwen3:8b")
            .unwrap_err();
        assert!(matches!(err, ResolveError::Unknown { .. }));
    }

    #[test]
    fn qualified_id_on_unreachable_server_passes_through() {
        // unknown upstream model name, no allowlist: let the proxy 502 later
        let r = registry().resolve(Flavor::OpenAi, "down/anything").unwrap();
        assert_eq!(r.upstream_model, "anything");
    }

    #[test]
    fn unknown_model_lists_known_ids() {
        let err = registry().resolve(Flavor::OpenAi, "nope").unwrap_err();
        match err {
            ResolveError::Unknown { known, .. } => {
                assert_eq!(known, vec!["studio/gpt-oss-20b".to_string()])
            }
            _ => panic!("expected Unknown"),
        }
    }

    #[test]
    fn bare_model_resolves_when_unique() {
        let r = registry().resolve(Flavor::Ollama, "qwen3:8b").unwrap();
        assert_eq!(r.server_url, "http://rig.lan:8080");
    }

    #[test]
    fn bare_model_ambiguous_lists_candidates() {
        let reg = registry();
        reg.set_models("studio", Some(vec![dm("shared")]));
        reg.set_models("rig", Some(vec![dm("shared")]));
        // need an openai+ollama pair is not ambiguous across flavors; add second ollama:
        reg.upsert_server(srv("rig2", Flavor::Ollama), Origin::State)
            .unwrap();
        reg.set_models("rig2", Some(vec![dm("shared")]));
        let err = reg.resolve(Flavor::Ollama, "shared").unwrap_err();
        match err {
            ResolveError::Ambiguous { candidates, .. } => {
                assert!(candidates.contains(&"rig/shared".to_string()));
                assert!(candidates.contains(&"rig2/shared".to_string()));
            }
            _ => panic!("expected Ambiguous"),
        }
    }

    #[test]
    fn remove_config_origin_server_is_rejected() {
        let err = registry().remove_server("studio").unwrap_err();
        assert!(matches!(err, RegistryError::ConfigOrigin(n) if n == "studio"));
    }

    #[test]
    fn upsert_then_remove_state_server() {
        let reg = registry();
        reg.upsert_server(srv("extra", Flavor::OpenAi), Origin::State)
            .unwrap();
        assert_eq!(reg.state_servers().len(), 1);
        let removed = reg.remove_server("extra").unwrap();
        assert_eq!(removed.config.name, "extra");
        assert!(reg.state_servers().is_empty());
    }

    #[test]
    fn list_models_only_includes_matching_flavor_and_reachable() {
        let reg = registry();
        let openai = reg.list_models(Flavor::OpenAi);
        assert_eq!(openai.len(), 1);
        assert_eq!(openai[0].0, "studio");
        let ollama = reg.list_models(Flavor::Ollama);
        assert_eq!(ollama.len(), 1);
        assert_eq!(ollama[0].1.upstream_name, "qwen3:8b");
    }
}
