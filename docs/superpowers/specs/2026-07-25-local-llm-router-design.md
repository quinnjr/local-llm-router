# local-llm-router — Design

**Date:** 2026-07-25
**Status:** Approved (pending user spec review)

A locally running, minimalist Rust server that routes OpenAI-compatible and
Ollama-native client connections to one or more LLM servers on a network
(llama.cpp, Ollama, vLLM, LM Studio, …). It exposes a unified model list per
API surface so clients can switch models with their existing model pickers.

## Goals

- Single static binary; one TOML config file; no database, no auth, no metrics.
- Pure pass-through proxy: zero protocol translation, minimal request
  modification, full streaming support.
- Two client surfaces: OpenAI-compatible (`/v1/...`) and Ollama-native
  (`/api/...`).
- Model discovery from backends; unified, namespaced model listing so any
  client can enumerate and switch models (`GET /v1/models`, `GET /api/tags`).
- Servers definable via config file **and** a runtime admin API (persisted).
- Duplicate model identity is a hard error, never silently resolved.

## Non-goals (YAGNI)

- Protocol translation (OpenAI↔Ollama). Surfaces are strictly separated.
- Load balancing / failover across backends.
- API keys, rate limiting, request logging beyond basic tracing.
- Virtual/aggregate models (e.g. fan-out "Council of Experts" endpoints).
  The naming scheme exists to *enable* council-style client code, not to
  implement it server-side.

## Architecture

**Dual pass-through.** No translation layer. Each registered backend has an
explicit API flavor, and each inbound surface routes only to backends of the
matching flavor:

```
clients                          router                     backends
─────────────────────────────────────────────────────────────────────────
OpenAI client ──► /v1/chat/completions ─┐
                 /v1/completions        ├─► flavor=openai ─► llama.cpp, vLLM,
                 /v1/embeddings         │                    LM Studio, …
                 GET /v1/models (merged)┘

Ollama client ──► /api/generate ────────┐
                 /api/chat              ├─► flavor=ollama ─► Ollama
                 GET /api/tags (merged) ┘
```

Consequence: an OpenAI client cannot reach a model on an `ollama`-flavor
server. Escape hatch without translation machinery: Ollama serves an
OpenAI-compatible `/v1` on the same port, so the same box may be registered a
second time under a different `name` with `flavor = "openai"`.

### Model identity

Every server has a required unique `name`. Exposed model IDs are namespaced:

```
<server-name>/<upstream-model-name>     e.g. studio/qwen3:8b, rig/gpt-oss-20b
```

This makes the same underlying LLM on two machines addressable as distinct
models (the "Council of Experts" use case) and makes collisions impossible
unless configuration is genuinely ambiguous.

**Collision rules — always hard errors:**

- Two servers with the same `name` → startup fails / admin API returns 409.
- Server names must not contain `/` (validated at load/add time → startup
  fails / 409). With unique, slash-free names, exposed IDs are unique *by
  construction*; the registry still asserts uniqueness on every discovery
  refresh as a defensive check.

**Bare-name matching:** an inbound `model` containing no `/` is resolved
against the surface's flavor: exactly one match → route; zero → 404; more
than one → 400 listing the fully qualified candidates.

### Request handling

1. Match exposed model ID → `(server, upstream_model_name)`.
2. Rewrite the `model` field in the request body to the upstream name.
   (The only request modification; without it Ollama upstreams reject
   namespaced IDs.)
3. Proxy to the same path on the backend (`/v1/chat/completions` →
   `{url}/v1/chat/completions`, `/api/chat` → `{url}/api/chat`).
4. Stream the response back unmodified — SSE or NDJSON byte stream, no
   buffering, no response rewriting (the response's `model` field stays the
   upstream name; clients ignore this in practice).
5. Client disconnect cancels the upstream request (tokio cancellation).

### Model discovery

- `flavor = "openai"` → `GET {url}/v1/models`
- `flavor = "ollama"` → `GET {url}/api/tags`

Discovery runs at startup, when a server is added via the admin API, and on
demand via `POST /admin/refresh`. Results are cached; `/v1/models` and
`/api/tags` are served from the cache (merged across servers of the
respective flavor).

A backend that is down at discovery time does **not** fail startup (home
networks have sleeping machines). It is marked unreachable with an empty
model list until the next refresh.

## Configuration

`config.toml` (path via CLI flag, default `./config.toml`):

```toml
[router]
listen = "0.0.0.0:4000"
request_timeout_secs = 600      # generous; LLMs are slow
# state_file = "./state.toml"   # default

[[server]]
name = "studio"                  # required, unique
url = "http://192.168.1.10:8080" # required
flavor = "openai"                # required: "openai" | "ollama"
# models = ["qwen3:8b"]          # optional allowlist of *upstream* model names; default: expose all discovered

[[server]]
name = "rig"
url = "http://192.168.1.11:11434"
flavor = "ollama"
```

## API surface

### Client-facing

| Endpoint | Behavior |
|---|---|
| `GET /v1/models` | Merged list from `openai`-flavor servers: upstream entries with `id` rewritten to the namespaced ID, `owned_by` set to the server name |
| `POST /v1/chat/completions` | Routed, streamed |
| `POST /v1/completions` | Routed, streamed |
| `POST /v1/embeddings` | Routed |
| `GET /api/tags` | Merged list from `ollama`-flavor servers: upstream entries with `name`/`model` rewritten to the namespaced ID |
| `POST /api/generate` | Routed, streamed |
| `POST /api/chat` | Routed, streamed |
| `GET /health` | Static 200 |

Unknown path on either surface → 404.

### Admin

| Endpoint | Behavior |
|---|---|
| `GET /admin/servers` | Servers with flavor, discovered models, reachability |
| `POST /admin/servers` | Add server (`name`, `url`, `flavor` required). 409 on duplicate name or `/` in name. Triggers discovery for the new server |
| `DELETE /admin/servers/{name}` | Remove. 409 if the server is defined in `config.toml` (edit the file instead) |
| `POST /admin/refresh` | Re-run discovery on all servers |

## State & persistence

Runtime-added servers are written to `state.toml` (same `[[server]]` schema).
Startup order: load `config.toml`, then overlay `state.toml`; names must be
unique across both (collision → startup error). The admin API mutates
`state.toml` only — hand-edited config is never rewritten by the server.

## Error handling

Error bodies match the inbound surface's native shape (OpenAI error object on
`/v1`, `{"error": "..."}` on `/api`).

| Condition | Status |
|---|---|
| Unknown model | 404 + list of known IDs |
| Ambiguous bare model | 400 + qualified candidates |
| Malformed JSON body ("request body is not valid JSON") | 400 |
| Backend unreachable / connection failure before upstream response | 502 |
| Upstream failure mid-stream (status already committed) | stream truncated, logged |
| Duplicate name or `/` in name (admin add) | 409 |
| Delete of config-defined server | 409 |

## Components

Single crate, small modules with one job each:

- `config` — TOML load/validate, state file read/write.
- `registry` — server table + discovered-model cache; exposed-ID ↔
  `(server, upstream name)` resolution; bare-name matching; collision
  detection.
- `proxy` — generic streaming reverse proxy: model-field rewrite, path
  forwarding, cancellation-safe byte streaming.
- `api_openai` — `/v1` handlers (thin: registry lookup → proxy).
- `api_ollama` — `/api` handlers (same shape).
- `api_admin` — admin handlers.
- `discovery` — per-flavor model-list fetching and parsing.
- `main` — CLI args, tracing init, axum router assembly.

### Tech stack

Rust (edition 2024), `axum`, `tokio`, `reqwest` (rustls, streaming),
`serde`, `toml`, `tracing` + `tracing-subscriber`. No other dependencies.
(`futures-util` was added during audit reconciliation for mid-stream error
logging — it is already in the lock tree via reqwest and adds zero new
packages.)

## Testing

- **Unit:** config parsing/validation; registry resolution (namespaced, bare,
  ambiguous, unknown); collision detection; model-field rewrite.
- **Integration:** mock upstream (in-process axum/hyper server) asserting:
  correct upstream path, model-field rewrite, SSE/NDJSON streaming
  pass-through, merged `/v1/models` + `/api/tags`, admin CRUD + state-file
  persistence, 404/400/409/502 error paths, startup with unreachable
  backend.

## Future directions (explicitly out of scope)

Virtual/aggregate models (one exposed ID fanning out to N backends for
council-style querying), failover, optional bearer-token auth, OpenAI→Ollama
translation so OpenAI clients can reach ollama-flavor models without a
second registration.
