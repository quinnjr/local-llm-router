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
