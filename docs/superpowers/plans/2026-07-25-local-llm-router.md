# local-llm-router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dual pass-through LLM router from `docs/superpowers/specs/2026-07-25-local-llm-router-design.md`: a minimalist Rust server routing OpenAI-compatible and Ollama-native connections to network LLM backends, with namespaced model listing for client-side model switching.

**Architecture:** Pure pass-through proxy. Two inbound surfaces (`/v1` OpenAI, `/api` Ollama-native), each routing only to backends of the matching declared flavor. No protocol translation. The only request mutation is rewriting the `model` body field from the exposed namespaced ID (`<server>/<model>`) back to the upstream model name. Responses stream back untouched.

**Tech Stack:** Rust 1.99 (edition 2024), axum 0.8, tokio, reqwest 0.12 (rustls, streaming), serde/serde_json, toml 0.8, tracing.

**Spec:** `docs/superpowers/specs/2026-07-25-local-llm-router-design.md` (commit db64bab)

## Global Constraints

- Rust edition **2024** (already set in `Cargo.toml`). Toolchain: rustc 1.99.0-nightly.
- Allowed dependencies, nothing else: `axum`, `tokio`, `reqwest` (default-features = false, features = `["rustls-tls", "stream", "json"]`), `serde`, `serde_json`, `toml`, `tracing`, `tracing-subscriber`. **No dev-dependencies** (mock servers are built with axum itself).
- Exposed model IDs are `<server-name>/<upstream-model-name>`. Server names must be unique and must not contain `/`. Duplicate name or `/` in name is always a hard error (startup failure or HTTP 409).
- Surfaces are strictly separated: `/v1/*` routes only to `flavor = "openai"` servers, `/api/*` only to `flavor = "ollama"` servers. **No protocol translation.**
- The only request modification: rewrite the top-level `model` JSON field to the upstream model name. Responses are streamed back unmodified.
- Error body shape matches the surface: OpenAI `{"error": {"message", "type", "code"}}` on `/v1`, `{"error": "..."}` on `/api`.
- A backend that is down at discovery time must NOT fail startup; mark it unreachable with an empty model list.
- Hand-edited `config.toml` is never rewritten by the server; runtime-added servers persist to `state.toml` only.
- Every task ends with a commit. Run `cargo test` before each commit; all tests must pass.
- Commit messages: `feat: ...` / `test: ...` conventional style, matching repo history (`docs: design spec...`).

---

### Task 1: Scaffolding + config/state loading

**Files:**
- Modify: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `src/config.rs`

**Interfaces:**
- Consumes: nothing.
- Produces (used by Tasks 2, 3, 7, 8):
  ```rust
  // src/config.rs
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
  pub enum Flavor { OpenAi, Ollama }   // serde: "openai" | "ollama"

  #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
  pub struct ServerConfig {
      pub name: String,
      pub url: String,
      pub flavor: Flavor,
      pub models: Option<Vec<String>>, // upstream-name allowlist
  }

  #[derive(Clone, Debug, Deserialize)]
  pub struct RouterConfig {
      pub listen: String,              // default "0.0.0.0:4000"
      pub request_timeout_secs: u64,   // default 600
      pub state_file: PathBuf,         // default "./state.toml"
  }

  #[derive(Clone, Debug, Deserialize)]
  pub struct Config { pub router: RouterConfig, pub server: Vec<ServerConfig> }

  #[derive(Debug)]
  pub enum ConfigError { Io(std::io::Error), Parse(toml::de::Error), Invalid(String) }

  impl Config {
      pub fn load(path: &Path) -> Result<Config, ConfigError>;
      pub fn validate_servers(servers: &[ServerConfig]) -> Result<(), ConfigError>;
  }

  #[derive(Clone, Debug, Default, Serialize, Deserialize)]
  pub struct StateFile { pub server: Vec<ServerConfig> }

  impl StateFile {
      pub fn load(path: &Path) -> Result<StateFile, ConfigError>; // missing file => empty
      pub fn save(path: &Path, servers: &[ServerConfig]) -> Result<(), ConfigError>;
  }
  ```

- [ ] **Step 1: Set dependencies and write failing tests**

`Cargo.toml`:

```toml
[package]
name = "local-llm-router"
version = "0.1.0"
edition = "2024"

[dependencies]
axum = "0.8"
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "stream", "json"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
```

`src/lib.rs`:

```rust
pub mod config;
```

`src/config.rs` — start with the test module only (implementation in step 3):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
[router]
listen = "127.0.0.1:4000"

[[server]]
name = "studio"
url = "http://192.168.1.10:8080"
flavor = "openai"

[[server]]
name = "rig"
url = "http://192.168.1.11:11434"
flavor = "ollama"
models = ["qwen3:8b"]
"#;

    #[test]
    fn parses_valid_config_with_defaults() {
        let cfg: Config = toml::from_str(GOOD).unwrap();
        assert_eq!(cfg.router.listen, "127.0.0.1:4000");
        assert_eq!(cfg.router.request_timeout_secs, 600);
        assert_eq!(cfg.router.state_file, PathBuf::from("./state.toml"));
        assert_eq!(cfg.server.len(), 2);
        assert_eq!(cfg.server[0].flavor, Flavor::OpenAi);
        assert_eq!(cfg.server[1].models, Some(vec!["qwen3:8b".to_string()]));
    }

    #[test]
    fn rejects_duplicate_server_names() {
        let dup = r#"
[[server]]
name = "a"
url = "http://x:1"
flavor = "openai"
[[server]]
name = "a"
url = "http://y:2"
flavor = "ollama"
"#;
        let servers: Vec<ServerConfig> = toml::from_str::<TomlServers>(dup).unwrap().server;
        let err = Config::validate_servers(&servers).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(m) if m.contains("duplicate")));
    }

    #[test]
    fn rejects_slash_in_server_name() {
        let bad = r#"
[[server]]
name = "a/b"
url = "http://x:1"
flavor = "openai"
"#;
        let servers: Vec<ServerConfig> = toml::from_str::<TomlServers>(bad).unwrap().server;
        let err = Config::validate_servers(&servers).unwrap_err();
        assert!(matches!(err, ConfigError::Invalid(m) if m.contains('/')));
    }

    #[test]
    fn missing_state_file_loads_empty() {
        let path = std::env::temp_dir().join(format!("llr-nope-{}.toml", std::process::id()));
        let state = StateFile::load(&path).unwrap();
        assert!(state.server.is_empty());
    }

    #[test]
    fn state_file_round_trips() {
        let path = std::env::temp_dir().join(format!("llr-state-{}.toml", std::process::id()));
        let servers = vec![ServerConfig {
            name: "rig".into(),
            url: "http://192.168.1.11:11434".into(),
            flavor: Flavor::Ollama,
            models: None,
        }];
        StateFile::save(&path, &servers).unwrap();
        let loaded = StateFile::load(&path).unwrap();
        assert_eq!(loaded.server, servers);
        std::fs::remove_file(&path).ok();
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `src/config.rs` items (`Config`, `Flavor`, `TomlServers`, …) do not exist, compile errors.

- [ ] **Step 3: Implement config module**

Prepend to `src/config.rs` (above the test module):

```rust
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Flavor {
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "ollama")]
    Ollama,
}

impl fmt::Display for Flavor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Flavor::OpenAi => write!(f, "openai"),
            Flavor::Ollama => write!(f, "ollama"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub name: String,
    pub url: String,
    pub flavor: Flavor,
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

fn default_listen() -> String {
    "0.0.0.0:4000".to_string()
}
fn default_timeout() -> u64 {
    600
}
fn default_state_file() -> PathBuf {
    PathBuf::from("./state.toml")
}

#[derive(Clone, Debug, Deserialize)]
pub struct RouterConfig {
    #[serde(default = "default_listen")]
    pub listen: String,
    #[serde(default = "default_timeout")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_state_file")]
    pub state_file: PathBuf,
}

impl Default for RouterConfig {
    fn default() -> Self {
        RouterConfig {
            listen: default_listen(),
            request_timeout_secs: default_timeout(),
            state_file: default_state_file(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub router: RouterConfig,
    #[serde(default)]
    pub server: Vec<ServerConfig>,
}

/// Helper for tests and for parsing bare `[[server]]` documents.
#[derive(Deserialize)]
pub struct TomlServers {
    #[serde(default)]
    pub server: Vec<ServerConfig>,
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Parse(toml::de::Error),
    Invalid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Io(e) => write!(f, "io error: {e}"),
            ConfigError::Parse(e) => write!(f, "toml parse error: {e}"),
            ConfigError::Invalid(m) => write!(f, "invalid config: {m}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        ConfigError::Io(e)
    }
}

impl Config {
    pub fn load(path: &Path) -> Result<Config, ConfigError> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Config = toml::from_str(&text).map_err(ConfigError::Parse)?;
        Config::validate_servers(&cfg.server)?;
        Ok(cfg)
    }

    pub fn validate_servers(servers: &[ServerConfig]) -> Result<(), ConfigError> {
        let mut seen = std::collections::HashSet::new();
        for s in servers {
            if s.name.is_empty() {
                return Err(ConfigError::Invalid("server name must not be empty".into()));
            }
            if s.name.contains('/') {
                return Err(ConfigError::Invalid(format!(
                    "server name '{}' must not contain '/'",
                    s.name
                )));
            }
            if !seen.insert(s.name.clone()) {
                return Err(ConfigError::Invalid(format!(
                    "duplicate server name '{}'",
                    s.name
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StateFile {
    #[serde(default)]
    pub server: Vec<ServerConfig>,
}

impl StateFile {
    pub fn load(path: &Path) -> Result<StateFile, ConfigError> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(ConfigError::Parse),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(StateFile::default()),
            Err(e) => Err(ConfigError::Io(e)),
        }
    }

    pub fn save(path: &Path, servers: &[ServerConfig]) -> Result<(), ConfigError> {
        let file = StateFile {
            server: servers.to_vec(),
        };
        let text = toml::to_string_pretty(&file)
            .map_err(|e| ConfigError::Invalid(format!("cannot serialize state: {e}")))?;
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(path, text)?;
        Ok(())
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/lib.rs src/config.rs .gitignore src/main.rs
git commit -m "feat: config and state file loading with validation"
```

(`.gitignore` and the `cargo init` `src/main.rs` hello-world are untracked from the skeleton; this commit sweeps them in.)

---

### Task 2: Registry — server table and model resolution

**Files:**
- Create: `src/registry.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Consumes: `crate::config::{Flavor, ServerConfig}` (Task 1).
- Produces (used by Tasks 3–7):
  ```rust
  pub enum Origin { Config, State }

  pub struct DiscoveredModel { pub upstream_name: String, pub raw: serde_json::Value }

  pub struct ServerEntry {
      pub config: ServerConfig,
      pub origin: Origin,
      pub models: Option<Vec<DiscoveredModel>>, // None = unreachable
  }
  impl ServerEntry { pub fn reachable(&self) -> bool; }

  pub struct Resolved { pub server_url: String, pub upstream_model: String }

  pub enum ResolveError {
      Unknown { model: String, known: Vec<String> },
      Ambiguous { model: String, candidates: Vec<String> },
  }

  pub enum RegistryError { DuplicateName(String), InvalidName(String), UnknownServer(String), ConfigOrigin(String) }

  pub struct Registry { /* RwLock<HashMap<String, ServerEntry>> */ }
  impl Registry {
      pub fn new(servers: Vec<(ServerConfig, Origin)>) -> Result<Registry, RegistryError>;
      pub fn resolve(&self, flavor: Flavor, model: &str) -> Result<Resolved, ResolveError>;
      pub fn upsert_server(&self, config: ServerConfig, origin: Origin) -> Result<(), RegistryError>;
      pub fn remove_server(&self, name: &str) -> Result<ServerEntry, RegistryError>;
      pub fn set_models(&self, name: &str, models: Option<Vec<DiscoveredModel>>);
      pub fn list_models(&self, flavor: Flavor) -> Vec<(String, DiscoveredModel)>; // (server name, model)
      pub fn list_servers(&self) -> Vec<ServerEntry>;
      pub fn state_servers(&self) -> Vec<ServerConfig>;
  }

  pub fn exposed_id(server: &str, upstream: &str) -> String; // "server/upstream"
  ```

Resolution rules (from spec):

- Model contains `/` → split at first `/` into `(server_name, upstream)`. Server must exist with matching flavor. If the server has discovered models (`Some`), `upstream` must be in the list. If models is `None` (unreachable): allow if the server has no allowlist, or if `upstream` is in `server.config.models` allowlist — then forward and let the proxy fail with 502. Otherwise `Unknown` with all exposed IDs for that flavor.
- Model has no `/` (bare) → search all servers of the flavor with `Some` models containing it. Exactly 1 → resolve. 0 → `Unknown`. >1 → `Ambiguous` with qualified candidates.

- [ ] **Step 1: Write failing tests**

`src/registry.rs` (test module only):

```rust
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
        let r = registry().resolve(Flavor::OpenAi, "studio/gpt-oss-20b").unwrap();
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
        let err = registry().resolve(Flavor::OpenAi, "rig/qwen3:8b").unwrap_err();
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
        reg.upsert_server(srv("rig2", Flavor::Ollama), Origin::State).unwrap();
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
        reg.upsert_server(srv("extra", Flavor::OpenAi), Origin::State).unwrap();
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `src/registry.rs` not referenced from `lib.rs` / items missing (compile error).

- [ ] **Step 3: Implement registry**

Add to `src/lib.rs`:

```rust
pub mod registry;
```

Prepend to `src/registry.rs` (above tests):

```rust
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
    Unknown { model: String, known: Vec<String> },
    Ambiguous { model: String, candidates: Vec<String> },
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
                write!(f, "server '{n}' is defined in config.toml; edit the file to remove it")
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
            let entry = inner
                .get(name)
                .filter(|e| e.config.flavor == flavor);
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
        out.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.upstream_name.cmp(&b.1.upstream_name)));
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 12 tests in `registry::tests` + 5 from Task 1.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/registry.rs
git commit -m "feat: registry with namespaced model resolution and collision rules"
```

---

### Task 3: AppState + discovery

**Files:**
- Create: `src/discovery.rs`
- Modify: `src/lib.rs`
- Create: `tests/common/mod.rs`
- Create: `tests/discovery.rs`

**Interfaces:**
- Consumes: `config::{Config, Flavor, ServerConfig}` (Task 1), `registry::{DiscoveredModel, Origin, Registry}` (Task 2).
- Produces (used by Tasks 4–8):
  ```rust
  // src/lib.rs
  #[derive(Clone)]
  pub struct AppState {
      pub registry: std::sync::Arc<Registry>,
      pub http: reqwest::Client,
      pub state_file: std::path::PathBuf,
  }
  impl AppState {
      pub fn new(config: &Config, state_servers: Vec<ServerConfig>)
          -> Result<AppState, registry::RegistryError>;
  }

  // src/discovery.rs
  pub async fn fetch_models(http: &reqwest::Client, server: &ServerConfig)
      -> Result<Vec<DiscoveredModel>, reqwest::Error>;
  pub async fn refresh_all(state: &AppState) -> Vec<(String, bool)>; // (server name, reachable)
  pub async fn refresh_one(state: &AppState, name: &str);
  ```

Discovery details:
- `Flavor::OpenAi` → `GET {url}/v1/models`, parse `data[]`: `upstream_name = item["id"]`, `raw = item`.
- `Flavor::Ollama` → `GET {url}/api/tags`, parse `models[]`: `upstream_name = item["name"]`, `raw = item`.
- `server.url` is trimmed of trailing `/` before joining.
- After parsing, apply the `server.models` allowlist (upstream names) if set.
- `refresh_all` iterates servers; success → `set_models(name, Some(models))`, failure → `set_models(name, None)` and log with `tracing::warn!`.

- [ ] **Step 1: Write failing tests**

`tests/common/mod.rs`:

```rust
// Per-target helper usage varies; each test binary compiles its own copy.
#![allow(dead_code)]

use axum::{Json, Router, routing::get, routing::post};
use std::sync::{Arc, Mutex};

/// Spawn any router on an ephemeral port; returns the base URL.
pub async fn spawn(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

#[derive(Clone, Default)]
pub struct Capture {
    pub last_model: Arc<Mutex<Option<String>>>,
}

/// Mock llama.cpp-style server: GET /v1/models + POST /v1/chat/completions.
/// The POST captures the `model` field and replies with a two-chunk SSE body.
pub async fn mock_openai(models: Vec<&str>) -> (String, Capture) {
    let capture = Capture::default();
    let data: Vec<serde_json::Value> = models
        .iter()
        .map(|m| serde_json::json!({"id": m, "object": "model", "created": 0, "owned_by": "mock"}))
        .collect();
    let cap = capture.clone();
    let router = Router::new()
        .route(
            "/v1/models",
            get(move || {
                let data = data.clone();
                async move { Json(serde_json::json!({"object": "list", "data": data})) }
            }),
        )
        .route(
            "/v1/chat/completions",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                async move {
                    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    *cap.last_model.lock().unwrap() =
                        v["model"].as_str().map(|s| s.to_string());
                    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
                    (
                        [("content-type", "text/event-stream")],
                        sse.to_string(),
                    )
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture)
}

/// Mock Ollama server: GET /api/tags + POST /api/chat (NDJSON reply).
pub async fn mock_ollama(models: Vec<&str>) -> (String, Capture) {
    let capture = Capture::default();
    let entries: Vec<serde_json::Value> = models
        .iter()
        .map(|m| {
            serde_json::json!({
                "name": m, "model": m, "modified_at": "2026-07-25T00:00:00Z",
                "size": 1000, "digest": "abc", "details": {"family": "qwen"}
            })
        })
        .collect();
    let cap = capture.clone();
    let router = Router::new()
        .route(
            "/api/tags",
            get(move || {
                let entries = entries.clone();
                async move { Json(serde_json::json!({"models": entries})) }
            }),
        )
        .route(
            "/api/chat",
            post(move |body: axum::body::Bytes| {
                let cap = cap.clone();
                async move {
                    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
                    *cap.last_model.lock().unwrap() =
                        v["model"].as_str().map(|s| s.to_string());
                    let ndjson = "{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"done\":false}\n{\"done\":true}\n";
                    ([("content-type", "application/x-ndjson")], ndjson.to_string())
                }
            }),
        );
    let base = spawn(router).await;
    (base, capture)
}
```

`tests/discovery.rs`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `local_llm_router::AppState` / `discovery` do not exist (compile error).

- [ ] **Step 3: Implement AppState + discovery**

`src/lib.rs`:

```rust
pub mod config;
pub mod discovery;
pub mod registry;

use config::{Config, ServerConfig};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<registry::Registry>,
    pub http: reqwest::Client,
    pub state_file: PathBuf,
}

impl AppState {
    pub fn new(
        config: &Config,
        state_servers: Vec<ServerConfig>,
    ) -> Result<AppState, registry::RegistryError> {
        let mut servers: Vec<(ServerConfig, registry::Origin)> = config
            .server
            .iter()
            .cloned()
            .map(|s| (s, registry::Origin::Config))
            .collect();
        servers.extend(state_servers.into_iter().map(|s| (s, registry::Origin::State)));
        let registry = registry::Registry::new(servers)?;
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(
                config.router.request_timeout_secs,
            ))
            .build()
            .expect("reqwest client builder failed");
        Ok(AppState {
            registry: Arc::new(registry),
            http,
            state_file: config.router.state_file.clone(),
        })
    }
}
```

`src/discovery.rs`:

```rust
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 4 discovery tests + 17 earlier tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/discovery.rs tests/common/mod.rs tests/discovery.rs
git commit -m "feat: per-flavor model discovery with unreachable-backend tolerance"
```

---

### Task 4: Streaming proxy with model-field rewrite

**Files:**
- Create: `src/proxy.rs`
- Modify: `src/lib.rs`
- Create: `tests/proxy.rs`

**Interfaces:**
- Consumes: `AppState` (Task 3), `registry::Resolved` (Task 2).
- Produces (used by Tasks 5, 6):
  ```rust
  pub enum ForwardError {
      MissingModel,
      BadJson,
      Upstream(reqwest::Error),
  }

  /// Rewrites body["model"] to resolved.upstream_model, POSTs `body` to
  /// `{resolved.server_url}{path}`, and streams the response back with the
  /// upstream status code and content-type.
  pub async fn forward(
      state: &AppState,
      resolved: &Resolved,
      path: &str,
      body: axum::body::Bytes,
  ) -> Result<axum::response::Response, ForwardError>;

  /// Extracts the top-level "model" string from a JSON request body.
  pub fn extract_model(body: &axum::body::Bytes) -> Option<String>;
  ```

- [ ] **Step 1: Write failing tests**

`tests/proxy.rs`:

```rust
mod common;

use axum::body::Bytes;
use local_llm_router::config::{Config, RouterConfig};
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `local_llm_router::proxy` does not exist (compile error).

- [ ] **Step 3: Implement proxy**

Add to `src/lib.rs`:

```rust
pub mod proxy;
```

`src/proxy.rs`:

```rust
use crate::AppState;
use crate::registry::Resolved;
use axum::body::{Body, Bytes};
use axum::response::Response;

#[derive(Debug)]
pub enum ForwardError {
    MissingModel,
    BadJson,
    Upstream(reqwest::Error),
}

impl std::fmt::Display for ForwardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ForwardError::MissingModel => write!(f, "request body has no \"model\" field"),
            ForwardError::BadJson => write!(f, "request body is not valid JSON"),
            ForwardError::Upstream(e) => write!(f, "upstream request failed: {e}"),
        }
    }
}

impl std::error::Error for ForwardError {}

pub fn extract_model(body: &Bytes) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    v["model"].as_str().map(|s| s.to_string())
}

pub async fn forward(
    state: &AppState,
    resolved: &Resolved,
    path: &str,
    body: Bytes,
) -> Result<Response, ForwardError> {
    let mut json: serde_json::Value =
        serde_json::from_slice(&body).map_err(|_| ForwardError::BadJson)?;
    if json["model"].as_str().is_none() {
        return Err(ForwardError::MissingModel);
    }
    json["model"] = serde_json::Value::String(resolved.upstream_model.clone());

    let url = format!("{}{}", resolved.server_url.trim_end_matches('/'), path);
    let upstream = state
        .http
        .post(&url)
        .json(&json)
        .send()
        .await
        .map_err(ForwardError::Upstream)?;

    let status = upstream.status();
    let content_type = upstream
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();

    // Byte-stream pass-through. When the client disconnects, axum drops this
    // body, which drops the reqwest stream and cancels the upstream request.
    let body = Body::from_stream(upstream.bytes_stream());
    Ok(Response::builder()
        .status(status.as_u16())
        .header("content-type", content_type)
        .body(body)
        .expect("response builder failed"))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 5 proxy tests + 21 earlier tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/proxy.rs tests/proxy.rs
git commit -m "feat: streaming pass-through proxy with model-field rewrite"
```

---

### Task 5: OpenAI surface (`/v1`)

**Files:**
- Create: `src/api_openai.rs`
- Modify: `src/lib.rs`
- Create: `tests/openai_api.rs`

**Interfaces:**
- Consumes: `AppState` (Task 3), `registry::{exposed_id, Flavor, ResolveError, Resolved}` (Tasks 1–2), `proxy::{extract_model, forward, ForwardError}` (Task 4).
- Produces (used by Tasks 7–8):
  ```rust
  pub fn router() -> axum::Router<AppState>;
  ```
  Routes: `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/embeddings`.
- Shared flow for POSTs: `extract_model` → `registry.resolve(Flavor::OpenAi, model)` → `proxy::forward`. Errors render as `{"error": {"message": ..., "type": "invalid_request_error", "code": null}}`: unknown model → 404 (message lists known IDs), ambiguous → 400 (lists candidates), missing/bad model field → 400, upstream failure → 502.
- `GET /v1/models`: from `registry.list_models(Flavor::OpenAi)`, clone each `raw` entry, set `raw["id"] = exposed_id(server, upstream)` and `raw["owned_by"] = server name`; respond `{"object": "list", "data": [...]}` sorted by id.

- [ ] **Step 1: Write failing tests**

`tests/openai_api.rs`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `local_llm_router::api_openai` does not exist (compile error).

- [ ] **Step 3: Implement OpenAI surface**

Add to `src/lib.rs`:

```rust
pub mod api_openai;
```

`src/api_openai.rs`:

```rust
use crate::config::Flavor;
use crate::proxy::{self, ForwardError};
use crate::registry::{ResolveError, exposed_id};
use crate::AppState;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(proxy_post))
        .route("/v1/completions", post(proxy_post))
        .route("/v1/embeddings", post(proxy_post))
}

async fn list_models(State(state): State<AppState>) -> Response {
    let mut data: Vec<serde_json::Value> = state
        .registry
        .list_models(Flavor::OpenAi)
        .into_iter()
        .map(|(server, model)| {
            let mut raw = model.raw.clone();
            raw["id"] = serde_json::Value::String(exposed_id(&server, &model.upstream_name));
            raw["owned_by"] = serde_json::Value::String(server);
            raw
        })
        .collect();
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}

async fn proxy_post(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    body: Bytes,
) -> Response {
    let path = uri.path().to_string();
    let Some(model) = proxy::extract_model(&body) else {
        return error(StatusCode::BAD_REQUEST, "request body has no \"model\" field");
    };
    let resolved = match state.registry.resolve(Flavor::OpenAi, &model) {
        Ok(r) => r,
        Err(ResolveError::Unknown { known, .. }) => {
            return error(
                StatusCode::NOT_FOUND,
                &format!("unknown model '{model}'. known models: {}", known.join(", ")),
            );
        }
        Err(ResolveError::Ambiguous { candidates, .. }) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!(
                    "ambiguous model '{model}'. qualified candidates: {}",
                    candidates.join(", ")
                ),
            );
        }
    };
    match proxy::forward(&state, &resolved, &path, body).await {
        Ok(resp) => resp,
        Err(e) => {
            let status = match e {
                ForwardError::MissingModel | ForwardError::BadJson => StatusCode::BAD_REQUEST,
                ForwardError::Upstream(_) => StatusCode::BAD_GATEWAY,
            };
            error(status, &e.to_string())
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": {"message": message, "type": "invalid_request_error", "code": null}
        })),
    )
        .into_response()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 5 openai_api tests + 26 earlier tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/api_openai.rs tests/openai_api.rs
git commit -m "feat: OpenAI-compatible /v1 surface with merged model listing"
```

---

### Task 6: Ollama surface (`/api`)

**Files:**
- Create: `src/api_ollama.rs`
- Modify: `src/lib.rs`
- Create: `tests/ollama_api.rs`

**Interfaces:**
- Consumes: same as Task 5, with `Flavor::Ollama`.
- Produces (used by Task 8):
  ```rust
  pub fn router() -> axum::Router<AppState>;
  ```
  Routes: `GET /api/tags`, `POST /api/generate`, `POST /api/chat`.
- `GET /api/tags`: from `registry.list_models(Flavor::Ollama)`, clone each `raw`, set `raw["name"] = exposed_id(...)` and `raw["model"] = exposed_id(...)`; respond `{"models": [...]}` sorted by name.
- Error shape on this surface: `{"error": "..."}` (plain string), statuses identical to Task 5 (404 unknown / 400 ambiguous / 400 missing model / 502 upstream).

- [ ] **Step 1: Write failing tests**

`tests/ollama_api.rs`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `local_llm_router::api_ollama` does not exist (compile error).

- [ ] **Step 3: Implement Ollama surface**

Add to `src/lib.rs`:

```rust
pub mod api_ollama;
```

`src/api_ollama.rs`:

```rust
use crate::config::Flavor;
use crate::proxy::{self, ForwardError};
use crate::registry::{ResolveError, exposed_id};
use crate::AppState;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/tags", get(list_tags))
        .route("/api/generate", post(proxy_post))
        .route("/api/chat", post(proxy_post))
}

async fn list_tags(State(state): State<AppState>) -> Response {
    let mut models: Vec<serde_json::Value> = state
        .registry
        .list_models(Flavor::Ollama)
        .into_iter()
        .map(|(server, model)| {
            let mut raw = model.raw.clone();
            let id = exposed_id(&server, &model.upstream_name);
            raw["name"] = serde_json::Value::String(id.clone());
            raw["model"] = serde_json::Value::String(id);
            raw
        })
        .collect();
    models.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
    Json(serde_json::json!({"models": models})).into_response()
}

async fn proxy_post(
    State(state): State<AppState>,
    uri: axum::http::Uri,
    body: Bytes,
) -> Response {
    let path = uri.path().to_string();
    let Some(model) = proxy::extract_model(&body) else {
        return error(StatusCode::BAD_REQUEST, "request body has no \"model\" field");
    };
    let resolved = match state.registry.resolve(Flavor::Ollama, &model) {
        Ok(r) => r,
        Err(ResolveError::Unknown { known, .. }) => {
            return error(
                StatusCode::NOT_FOUND,
                &format!("unknown model '{model}'. known models: {}", known.join(", ")),
            );
        }
        Err(ResolveError::Ambiguous { candidates, .. }) => {
            return error(
                StatusCode::BAD_REQUEST,
                &format!(
                    "ambiguous model '{model}'. qualified candidates: {}",
                    candidates.join(", ")
                ),
            );
        }
    };
    match proxy::forward(&state, &resolved, &path, body).await {
        Ok(resp) => resp,
        Err(e) => {
            let status = match e {
                ForwardError::MissingModel | ForwardError::BadJson => StatusCode::BAD_REQUEST,
                ForwardError::Upstream(_) => StatusCode::BAD_GATEWAY,
            };
            error(status, &e.to_string())
        }
    }
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": message}))).into_response()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 3 ollama_api tests + 31 earlier tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/api_ollama.rs tests/ollama_api.rs
git commit -m "feat: Ollama-native /api surface with merged tag listing"
```

---

### Task 7: Admin API + state persistence + health

**Files:**
- Create: `src/api_admin.rs`
- Modify: `src/lib.rs`
- Create: `tests/admin_api.rs`

**Interfaces:**
- Consumes: `config::{ServerConfig, StateFile}` (Task 1), `registry::{Origin, RegistryError}` (Task 2), `discovery::refresh_all/refresh_one` (Task 3), `AppState` (Task 3).
- Produces (used by Task 8):
  ```rust
  pub fn router() -> axum::Router<AppState>;
  ```
  Routes: `GET /health` (static `"ok"`), `GET /admin/servers`, `POST /admin/servers`, `DELETE /admin/servers/{name}`, `POST /admin/refresh`.

Behavior:
- `GET /admin/servers` → 200 `{"servers": [{name, url, flavor, origin: "config"|"state", reachable, models: [exposed IDs]}]}` sorted by name.
- `POST /admin/servers` body = `ServerConfig` JSON (`{"name", "url", "flavor", "models"?}`) → validate: `name` non-empty and slash-free and unique (else 409), `url` parses via `reqwest::Url::parse` (else 400) → `registry.upsert_server(.., Origin::State)` → persist `StateFile::save(&state.state_file, &registry.state_servers())` (else 500) → `discovery::refresh_one(state, name)` → 201 with the server entry JSON (same shape as list item).
- `DELETE /admin/servers/{name}` → `ConfigOrigin` → 409; `UnknownServer` → 404; success → persist state file → 204.
- `POST /admin/refresh` → `discovery::refresh_all` → 200 `{"results": [{"name": ..., "reachable": ...}]}`.
- All admin errors are plain JSON `{"error": "..."}`.

- [ ] **Step 1: Write failing tests**

`tests/admin_api.rs`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `local_llm_router::api_admin` does not exist (compile error).

- [ ] **Step 3: Implement admin API**

Add to `src/lib.rs`:

```rust
pub mod api_admin;
```

`src/api_admin.rs`:

```rust
use crate::config::{ServerConfig, StateFile};
use crate::registry::{Origin, RegistryError, exposed_id};
use crate::{AppState, discovery};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/admin/servers", get(list_servers).post(add_server))
        .route("/admin/servers/{name}", delete(remove_server))
        .route("/admin/refresh", post(refresh))
}

fn server_json(entry: &crate::registry::ServerEntry) -> serde_json::Value {
    let models: Vec<String> = entry
        .models
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(|m| exposed_id(&entry.config.name, &m.upstream_name))
        .collect();
    serde_json::json!({
        "name": entry.config.name,
        "url": entry.config.url,
        "flavor": entry.config.flavor.to_string(),
        "origin": match entry.origin {
            Origin::Config => "config",
            Origin::State => "state",
        },
        "reachable": entry.reachable(),
        "models": models,
    })
}

async fn list_servers(State(state): State<AppState>) -> Response {
    let servers: Vec<serde_json::Value> = state
        .registry
        .list_servers()
        .iter()
        .map(server_json)
        .collect();
    Json(serde_json::json!({"servers": servers})).into_response()
}

fn persist(state: &AppState) -> Result<(), Response> {
    StateFile::save(&state.state_file, &state.registry.state_servers()).map_err(|e| {
        error(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("failed to persist state file: {e}"),
        )
    })
}

async fn add_server(
    State(state): State<AppState>,
    Json(config): Json<ServerConfig>,
) -> Response {
    if config.name.is_empty() || config.name.contains('/') {
        return error(
            StatusCode::CONFLICT,
            "server name must not be empty or contain '/'",
        );
    }
    if reqwest::Url::parse(&config.url).is_err() {
        return error(StatusCode::BAD_REQUEST, "invalid url");
    }
    let name = config.name.clone();
    if let Err(e) = state.registry.upsert_server(config, Origin::State) {
        let status = match e {
            RegistryError::DuplicateName(_) | RegistryError::InvalidName(_) => {
                StatusCode::CONFLICT
            }
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        return error(status, &e.to_string());
    }
    if let Err(resp) = persist(&state) {
        let _ = state.registry.remove_server(&name);
        return resp;
    }
    discovery::refresh_one(&state, &name).await;
    let entry = state
        .registry
        .list_servers()
        .into_iter()
        .find(|e| e.config.name == name)
        .expect("just inserted");
    (StatusCode::CREATED, Json(server_json(&entry))).into_response()
}

async fn remove_server(State(state): State<AppState>, Path(name): Path<String>) -> Response {
    if let Err(e) = state.registry.remove_server(&name) {
        let status = match e {
            RegistryError::UnknownServer(_) => StatusCode::NOT_FOUND,
            RegistryError::ConfigOrigin(_) => StatusCode::CONFLICT,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        return error(status, &e.to_string());
    }
    if let Err(resp) = persist(&state) {
        return resp;
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn refresh(State(state): State<AppState>) -> Response {
    let results = discovery::refresh_all(&state).await;
    let results: Vec<serde_json::Value> = results
        .into_iter()
        .map(|(name, reachable)| serde_json::json!({"name": name, "reachable": reachable}))
        .collect();
    Json(serde_json::json!({"results": results})).into_response()
}

fn error(status: StatusCode, message: &str) -> Response {
    (status, Json(serde_json::json!({"error": message}))).into_response()
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test`
Expected: PASS — 8 admin tests + 34 earlier tests.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/api_admin.rs tests/admin_api.rs
git commit -m "feat: admin API with runtime server management and state persistence"
```

---

### Task 8: main.rs wiring + end-to-end test

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/main.rs`
- Create: `tests/e2e.rs`
- Create: `config.example.toml`

**Interfaces:**
- Consumes: everything.
- Produces:
  ```rust
  // src/lib.rs
  pub fn build_router(state: AppState) -> axum::Router;
  ```
  Merges `api_openai::router()`, `api_ollama::router()`, `api_admin::router()` and applies `.with_state(state)`.
- `main.rs`: parse `--config PATH` (default `./config.toml`), init tracing (`tracing_subscriber` fmt with env-filter, default level `info`), `Config::load` + `StateFile::load` + `AppState::new`, `discovery::refresh_all`, bind `router.listen`, serve. Startup errors print to stderr and exit non-zero.

- [ ] **Step 1: Write failing end-to-end test**

`tests/e2e.rs`:

```rust
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
```

`config.example.toml`:

```toml
[router]
listen = "0.0.0.0:4000"
request_timeout_secs = 600
# state_file = "./state.toml"

# A llama.cpp / vLLM / LM Studio server (OpenAI-compatible)
[[server]]
name = "studio"
url = "http://192.168.1.10:8080"
flavor = "openai"

# An Ollama server
[[server]]
name = "rig"
url = "http://192.168.1.11:11434"
flavor = "ollama"
# models = ["qwen3:8b"]  # optional upstream-name allowlist
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test`
Expected: FAIL — `local_llm_router::build_router` does not exist (compile error).

- [ ] **Step 3: Implement build_router and main**

Add to `src/lib.rs`:

```rust
pub fn build_router(state: AppState) -> axum::Router {
    axum::Router::new()
        .merge(api_openai::router())
        .merge(api_ollama::router())
        .merge(api_admin::router())
        .with_state(state)
}
```

Replace `src/main.rs` entirely:

```rust
use local_llm_router::config::{Config, StateFile};
use local_llm_router::{AppState, build_router, discovery};
use std::path::PathBuf;
use std::process::exit;

fn parse_config_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut path = PathBuf::from("./config.toml");
    while let Some(arg) = args.next() {
        if arg == "--config" {
            match args.next() {
                Some(p) => path = PathBuf::from(p),
                None => {
                    eprintln!("--config requires a path argument");
                    exit(2);
                }
            }
        } else if !arg.starts_with('-') {
            path = PathBuf::from(arg);
        }
    }
    path
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config_path = parse_config_path();
    let config = match Config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to load {}: {e}", config_path.display());
            exit(1);
        }
    };
    let state_file = match StateFile::load(&config.router.state_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "failed to load state file {}: {e}",
                config.router.state_file.display()
            );
            exit(1);
        }
    };
    let state = match AppState::new(&config, state_file.server) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("invalid server configuration: {e}");
            exit(1);
        }
    };

    let results = discovery::refresh_all(&state).await;
    for (name, reachable) in &results {
        if !reachable {
            tracing::warn!(server = %name, "server unreachable at startup; will retry on refresh");
        }
    }

    let app = build_router(state);
    let listener = match tokio::net::TcpListener::bind(&config.router.listen).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("failed to bind {}: {e}", config.router.listen);
            exit(1);
        }
    };
    tracing::info!(listen = %config.router.listen, "local-llm-router up");
    if let Err(e) = axum::serve(listener, app).await {
        eprintln!("server error: {e}");
        exit(1);
    }
}
```

- [ ] **Step 4: Run full test suite and a manual smoke test**

Run: `cargo test`
Expected: PASS — 1 e2e test + 42 earlier tests.

Run a real smoke test of the binary against a mock upstream:

```bash
cargo build
cat > /tmp/opencode/llr-smoke.toml <<'EOF'
[router]
listen = "127.0.0.1:4100"
EOF
./target/debug/local-llm-router --config /tmp/opencode/llr-smoke.toml &
sleep 1
curl -s http://127.0.0.1:4100/health
curl -s http://127.0.0.1:4100/v1/models
curl -s http://127.0.0.1:4100/admin/servers
kill %1
```

Expected: `ok`, an empty model list `{"object":"list","data":[]}`, and an empty servers list — proving the binary boots, binds, and serves with zero configured backends.

- [ ] **Step 5: Commit**

```bash
git add src/lib.rs src/main.rs tests/e2e.rs config.example.toml
git commit -m "feat: wire up binary with CLI, tracing, and startup discovery"
```

---

## Self-Review Notes

- **Spec coverage:** config+state (T1), namespacing/collisions/bare-match (T2), discovery + unreachable tolerance (T3), model rewrite + streaming + cancellation (T4), `/v1` surface (T5), `/api` surface (T6), admin CRUD + persistence + health + refresh (T7), CLI/startup/error exits (T8). All spec error codes (400/404/409/502) are exercised in tests.
- **Type consistency:** `ResolveError`, `RegistryError`, `ForwardError`, `AppState::new(&Config, Vec<ServerConfig>)`, `Registry::new(Vec<(ServerConfig, Origin)>)`, `refresh_all(&AppState) -> Vec<(String, bool)>`, `router() -> Router<AppState>` signatures match across tasks.
- **Placeholder scan:** all steps contain complete code; no TODOs.
