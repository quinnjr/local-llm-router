# local-llm-router

A locally running, minimalist server that routes OpenAI-compatible and
Ollama-native client connections to one or more LLM servers on your network
(llama.cpp, Ollama, vLLM, LM Studio, …). It exposes a unified model list per
API surface so clients can switch models with their existing model pickers.

Single static binary. One TOML config file. No database, no auth, no metrics.

## Features

- **Dual pass-through proxy** — no protocol translation. `/v1/*` routes to
  `flavor = "openai"` backends, `/api/*` to `flavor = "ollama"` backends.
- **Namespaced model identity** — every backend has a name; exposed model IDs
  are `<server>/<model>` (e.g. `studio/gpt-oss-20b`), so the same LLM on two
  machines is addressable as distinct models (Council-of-Experts style).
  Duplicate or `/`-containing server names are hard errors.
- **Unified model lists** — `GET /v1/models` and `GET /api/tags` merge
  per-flavor discovered models; bare model names resolve when unambiguous.
- **Config + runtime management** — `[[server]]` TOML blocks and an admin API
  persisted to `state.toml`.
- **Home-network tolerant** — down backends never fail startup (5s connect
  timeout vs. 600s streaming timeout); responses stream untouched with
  client-disconnect cancellation.

## Install

```sh
cargo install local-llm-router
```

Or build from source: `cargo build --release` (binary at `target/release/local-llm-router`).

## Quick start

```sh
cp config.example.toml config.toml   # edit your servers
local-llm-router                     # or: local-llm-router --config /path/to/config.toml
```

```toml
[router]
listen = "0.0.0.0:4000"

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
```

Point any OpenAI client at `http://localhost:4000/v1` or any Ollama client at
`http://localhost:4000` — models appear as `studio/gpt-oss-20b`,
`rig/qwen3:8b`, etc.

## API surface

| Endpoint | Behavior |
|---|---|
| `GET /v1/models` | Merged model list from `openai` backends |
| `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | Routed, streamed |
| `GET /api/tags` | Merged model list from `ollama` backends |
| `POST /api/generate`, `/api/chat` | Routed, streamed |
| `GET /admin/servers` · `POST /admin/servers` · `DELETE /admin/servers/{name}` | Runtime backend management (persisted to `state.toml`) |
| `POST /admin/refresh` | Re-run model discovery on all backends |
| `GET /health` | Static 200 |

## Security note

The admin API is unauthenticated by design: anyone who can reach the port can
add/remove backends. On untrusted networks, bind `127.0.0.1` or firewall the
port.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
