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

Verify the server is running:
```bash
curl http://127.0.0.1:8080/health
```

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

---

## ⚖️ License & Disclaimer

MIT License. This project is intended for developer personal research, educational experiments, and interoperability testing. Not affiliated with or endorsed by Xiaomi Inc.
