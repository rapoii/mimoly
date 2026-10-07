# Mimoly 🌸

> **Ultra-Lightweight, 100% Pure HTTP OpenAI-Compatible Web2API for Xiaomi MiMo Studio.**  
> 0% Chrome processes · 0 byte background browser · 0 CDP overhead · ~30 MB RAM.  
> 100% Free · No API keys · OpenAI SSE Streaming & Autonomous ReAct Tool Calling.

Mimoly bridges your existing web session on [Xiaomi MiMo Studio](https://aistudio.xiaomimimo.com) into a standard OpenAI REST API (`http://127.0.0.1:8080/v1`). It runs **completely without a browser in the background**, translating SSE streams directly over pure HTTP to power autonomous coding agents like **Hermes Agent, OpenCode, Claude Code, Cline, and 9router**.

---

## ✨ Features

- **True Web2API Architecture**: 100% pure asynchronous HTTP (`FastAPI` + `httpx`). Zero Chrome/Edge instances, zero CDP connections, zero temporary user profiles.
- **Ultra Low Memory Footprint**: Uses only ~30–50 MB RAM, compared to 400MB–1.2GB for headless browser automation.
- **Autonomous Tool Calling (ReAct Emulation)**: Native multi-tool calling support (terminal/shell execution, file reads/writes, patch, search) with bidirectional format mapping for AI coding agents.
- **OpenAI-Standard SSE Streaming**: Emits compliant `text/event-stream` chunks (`data: {"choices": [{"delta": ...}]}\n\n`) compatible with Vercel AI SDK, official `openai` Python/Node SDKs, and CLI tools.
- **Full Usage & Reasoning Metrics**: Accurately maps upstream `nativeUsage` tokens (`prompt_tokens`, `completion_tokens`, `reasoning_tokens`).
- **Clean Thinking Filter**: Automatically strips upstream `<think>` delimiter artifacts to prevent client prefill deadlocks.
- **Multi-Client Routing Support**: Exposes OpenAI and Ollama-compatible endpoints (`/v1/chat/completions`, `/v1/models`, `/v1/models/{id}`, `/api/tags`, `/api/show`).

---

## 🚀 Quick Start

### 1. Requirements & Installation

```bash
git clone https://github.com/rapoii/mimoly.git
cd mimoly

# Using uv (recommended)
uv venv
uv pip install -r requirements.txt

# Or using standard pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Session (`session.json`)

Export or copy your active Xiaomi MiMo session cookies into `session.json` in the root folder:

```json
[
  {
    "name": "userId",
    "value": "YOUR_USER_ID",
    "domain": ".xiaomimimo.com",
    "path": "/"
  },
  {
    "name": "xiaomichatbot_serviceToken",
    "value": "YOUR_SERVICE_TOKEN",
    "domain": ".xiaomimimo.com",
    "path": "/"
  },
  {
    "name": "xiaomichatbot_ph",
    "value": "YOUR_XIAOMICHATBOT_PH",
    "domain": ".xiaomimimo.com",
    "path": "/"
  }
]
```

> **Note**: `session.json` is already excluded in `.gitignore` to protect your credentials. See `session.example.json` for reference.

### 3. Run Proxy Server

```bash
# Start background-ready proxy on port 8080
python mimoly.py serve --port 8080

# Or with uv
uv run python mimoly.py serve --port 8080
```

#### Optional Flags & Environment Variables
- `--session-file <path>` (or `MIMOLY_SESSION_FILE`): Custom path to your `session.json`.
- `--workspace <path>` (or `MIMOLY_WORKSPACE`): Optional workspace root directory to anchor relative file paths for agent tools. By default, relative paths are preserved as-is.

#### Reliability & Scale (multi-account, overload protection)

All optional — sensible defaults keep a single-account deployment working unchanged.

| Env var | Default | Purpose |
| --- | --- | --- |
| `MIMOLY_ACCOUNTS_FILE` | *(unset)* | Path to a JSON list of accounts `[{"id": "...", "cookies": {...}}, ...]`. Enables multi-account rotation; falls back to `session.json` when unset. |
| `MIMOLY_MAX_INFLIGHT` | `0` (unlimited) | Max concurrent in-flight requests. Excess returns a fast **503** (`Retry-After`) instead of queueing unboundedly. |
| `MIMOLY_CACHE_TTL` | `300` | TTL (seconds) of the exact-match cache for **non-streaming** requests. `0` disables. |
| `MIMOLY_CACHE_MAXSIZE` | `256` | Max cached responses (LRU). |
| `MIMOLY_MAX_CONNECTIONS` | `200` | httpx connection-pool size (shared client, keep-alive reused across requests). |
| `MIMOLY_ACCOUNT_MAX_FAILURES` | `3` | Consecutive failures before an account's circuit breaker opens. |
| `MIMOLY_ACCOUNT_COOLDOWN_BASE` / `_MAX` | `30` / `1800` | Exponential per-account cooldown (seconds), reset on success. |

**Behaviour:** requests pick the least-busy healthy account; a `401/403` parks that
account and rotates to the next one pre-stream; `429/5xx` retry the same key with
full-jitter backoff inside a wall-clock budget; retries never happen after the
first streamed byte. `/health` and `/v1/stats` expose live pool/admission/cache state.

Verify the server is running:
```bash
curl http://127.0.0.1:8080/health
```

### 📊 Live Dashboard & Stats

A single-file, auto-refreshing dashboard is served at `/dashboard`
(refreshes every 2s, no build step). It uses a **neobrutalism** visual theme
(thick ink borders, hard offset shadows, flat high-contrast colors) and renders
live charts via **Chart.js** (loaded from CDN, with a graceful offline fallback):

```bash
# Open in browser
start http://127.0.0.1:8080/dashboard     # Windows
open  http://127.0.0.1:8080/dashboard     # macOS

# Raw JSON (same data)
curl http://127.0.0.1:8080/v1/stats
```

**Charts:** finish-reason doughnut, token-split doughnut (input/output/reasoning),
tool-usage bars, tokens-per-model bars, and a dual-axis activity timeline (last 60 min).

**Recent requests log:** the bottom panel shows the last 50 requests one by one —
each row has the finish-reason / model / stream chips, a timestamp, and side-by-side
**▲ Input** (the user prompt) and **▼ Output** (the model's reply) blocks, plus
latency, TTFT, and token counts. It is also exposed as JSON under `recent_requests`.

Tracked metrics (all in-memory, reset on restart):

| Metric | Description |
|---|---|
| `requests` / `errors` | total proxied requests and failures |
| `tool_calls` | total tool calls emitted by the model |
| `coerced_args` | tool args auto-repaired by `coerce_tool_args` |
| `invalid_args` | tool args still mismatched after coercion |
| `tokens.prompt_tokens` | cumulative **input** tokens |
| `tokens.completion_tokens` | cumulative **output** tokens |
| `tokens.total_tokens` | cumulative input + output |
| `tokens.reasoning_tokens` | reasoning ("thinking") tokens |
| `finish_reasons` | how requests ended (`stop` / `tool_calls` / `error`) |
| `latency` | end-to-end latency `{count, avg_ms, min_ms, max_ms}` |
| `ttft` | time-to-first-token for streaming `{count, avg_ms, min_ms, max_ms}` |
| `timeline` | per-minute buckets of `{requests, errors, tool_calls, total_tokens}` |
| `recent_requests` | last 50 requests: `{id, ts, model, stream, input, output, finish_reason, latency_ms, ttft_ms, prompt_tokens, completion_tokens, tools_called}` |
| `tool_usage` | per-tool call counts (top 20) |
| `model_tokens` | per-model token breakdown |

---

## 📦 Supported Models & Thinking Controls

Mimoly exclusively provides Xiaomi's two premier models with **maximum reasoning effort (mentok / rata kanan)** enabled by default:

| Model ID | Architecture | Description |
|---|---|---|
| `mimo-v2.6-pro` | Flagship Deep Reasoning | Xiaomi's most powerful reasoning model with full chain-of-thought analysis, edge-case evaluation, and complex tool calling. |
| `mimo-v2.6-flash` | High-Speed & Smart | Ultra-fast, highly responsive model with full reasoning capabilities for rapid coding and iterations. |

### 🧠 Thinking & Reasoning Effort Mentok

- **Default Maximum Effort**: Every request automatically runs at maximum thinking capacity ("rata kanan mentok"), instructing the reasoning engine to rigorously explore edge cases, formulate step-by-step hypotheses, and verify logic before outputting solutions.
- **Real-time Thinking Streaming**: When streaming, reasoning tokens are delivered live via standard OpenAI/DeepSeek format (`delta.reasoning_content`), allowing CLI interfaces (Hermes, OpenCode, Claude Code) and web clients (NextChat, Cherry Studio) to render live thought processes.
- **Thinking Toggle**: Pass `"enable_thinking": false` or `"reasoning_effort": "none"` to suppress reasoning tokens if pure instant output is preferred.

---

## 🔌 Client Integrations

### Hermes Agent CLI

Add the custom provider to `~/.hermes/config.yaml`:

```yaml
providers:
  custom:
    mimoly:
      base_url: "http://127.0.0.1:8080/v1"
      api_key: "no-key"

model: "mimoly/mimo-v2.6-pro"
```

Run Hermes with autonomous tools:
```bash
hermes chat -q "Baca file requirements.txt dan jelaskan dependensinya" --provider mimoly -m mimo-v2.6-pro
```

Or with high-speed flash:
```bash
hermes chat -q "Tulis script kalkulator CLI sederhana" --provider mimoly -m mimo-v2.6-flash
```

### Universal Agent Framework Adapter (`mimoly-cli`)

**Use Mimoly as backend for ANY agent CLI without modifying the CLI itself** (Hermes, Claude Code, Codex, OpenCode, Pi, Oh-My-Pi, etc).

The wrapper auto-detects which CLI you invoke, sets the `MIMOLY_FRAMEWORK` env var, and re-execs the upstream binary unchanged. No patches, no forks.

```bash
# Launch any CLI through the wrapper
mimoly-cli hermes chat -q "tulis function X"
mimoly-cli claude "refactor file src/app.tsx"
mimoly-cli codex "implement binary search in Python"
mimoly-cli opencode "add dark mode"
mimoly-cli pi "explain this regex"

# Override framework explicitly
MIMOLY_FRAMEWORK=claude-code mimoly-cli some-other-cli

# Or hit Mimoly directly with header (any HTTP client)
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "X-Agent-Framework: claude-code" \
  -H "Content-Type: application/json" \
  -d '{"model": "mimo-v2.6-pro", "messages": [...]}'
```

Supported frameworks (each gets a tailored tool-call prompt template):

| Framework | CLI Binaries | Tool Call Format |
|---|---|---|
| `default` | (fallback) | Qwen XML `<function=...>` |
| `claude-code` | `claude`, `claude-code` | Anthropic XML/JSON hybrid |
| `codex` | `codex` | Strict OpenAI JSON function calling |
| `opencode` | `opencode`, `oc` | XML or JSON markdown (flexible) |
| `hermes` | `hermes` | Qwen XML strict |
| `pi` | `pi`, `oh-my-pi` | Simple XML |

Resolution order: `X-Agent-Framework` header > body `agent_framework` > `MIMOLY_FRAMEWORK` env > fallback to `default`.

---

## 🩺 Troubleshooting: search & fetch via MCP playwright

Mimoly itself has no search tool. When Hermes runs with an MCP browser server
(e.g. `-t terminal,playwright`), two upstream/Hermes quirks can make it look like
mimoly is broken. Neither is.

### 1. A pasted link produced a summary instead of a tool call (fixed in the proxy)

Xiaomi MiMo Studio performs its **own** web retrieval whenever the prompt
contains a literal URL, then hands the page to the model as pre-fetched context.
The model reads "the page is already available" and answers from that payload
without ever calling `browser_navigate` — so a user who pastes a link gets a
summary and no tool call, even with a healthy toolset.

`webSearchStatus: "disabled"` in the upstream `modelConfig` does **not** suppress
it, and the retrieval happens server-side before generation, so there is no frame
to strip in the parser. Mimoly therefore defuses it at the source: user-turn URLs
are spaced out before the prompt goes upstream
(`https://en . wikipedia . org / wiki / Xiaomi`). The injector no longer
recognises a URL, while the model reads it fine and emits the **clean** url in the
tool call. No configuration needed — it is automatic.

Measured: raw URL `0/16` tool calls → masked `15/16`; end-to-end through Hermes
with a pasted link `0/6` → `6/6`.

### 2. `⚠️ Unknown toolset: playwright` and only `terminal` loaded

Hermes validates `-t terminal,playwright` against the toolset registry *before*
`wait_for_mcp_discovery()` returns. If the agent is built ahead of discovery
(~3 s for a warm `npx @playwright/mcp`), the toolset resolves to nothing, the
agent is built text-only, and the model honestly reports "no browser tools".

It is intermittent, so one green run proves nothing — score a pass **rate**. The
one-shot bound is `mcp_single_query_discovery_timeout` (default 15 s); raise it:

```bash
hermes config set mcp_single_query_discovery_timeout 45
```

A/B in the real CLI: bound `0.3` → 3/3 runs with `Unknown toolset` and 2 tools;
bound `45` → 3/3 runs with no warning and 26 tools. Leave
`mcp_discovery_timeout` (interactive, default 1.5 s) alone — an interactive
session recovers a late server on the next turn; a one-shot run has no second
turn, which is why only the single-query key needs room.

### 3. Search engines: prefer server-rendered pages

Bing/Google result pages render their list with JavaScript, so an accessibility
snapshot shows only chrome ("Short videos", "Related searches") and the model has
nothing to cite. Use Bing's RSS form or a static page:

```
https://www.bing.com/search?q=<query>&format=rss
```

DuckDuckGo may be unreachable from some networks (20 s timeout); Bing/Google
answer normally.

### 4. MCP tools never get called: JSON-type drift in tool arguments (fixed in the proxy)

Running a realistic multi-turn *vibe-coder* session with every toolset enabled
(see `examples/vibe_coder_turns.json` + `vibe_session.py`) showed the MCP servers
looking dead — the model called `tool_search` / `tool_describe` a few times, got
errors, gave up, and finished the whole task with `terminal`/`read_file`.

Root cause is the upstream model, not the proxy: MiMo emits some tool arguments
with the wrong JSON type. Reproduced 3/3 on a direct call:

```
tool_search arguments: {"queries": "[\"open browser\", \"take screenshot\"]"}   # string, not array
tool_call   arguments: {"calls": [{"tool": "mcp__playwright__browser_navigate"}]}  # wants "name"
```

Hermes' `tool_search` rejects the stringified array and `tool_call` rejects the
missing `name`, so the deferred-tool path collapses. Mimoly now repairs the
arguments against the declared schema (`coerce_tool_args`): a stringified array
becomes a real list, a bare string becomes a one-item list, `integer`/`boolean`
strings are converted, and array-of-object item keys are renamed
(`tool`/`function`/`id` → `name`). Streaming and non-streaming paths both use it.

Verified end-to-end after the fix — playwright really drives a browser:

```
mcp__playwright__browser_navigate {"url": "https://example.com"}
  → Page Title: Example Domain
```

Note the model still *prefers* `terminal` when ~250 tools are loaded at once;
the fix removes the *blocker*, it does not force tool choice. Give a task a
narrow toolset (`-t playwright`) when you want the MCP path used reliably.

### 5. Verify with the event stream, not the prose

The final answer can claim "no browser tools available" while the run actually
worked. Use `--format stream-json` and count `tool_use` events — that is the
ground truth. A ready-made harness lives in the repo:

```bash
.venv/Scripts/python.exe search_fetch_e2e.py --trials 3
```

---

## ⚖️ License & Disclaimer

MIT License. This project is intended for developer personal research, educational experiments, and interoperability testing. Not affiliated with or endorsed by Xiaomi Inc.
