#!/usr/bin/env python3
"""
Mimoly — 100% Pure HTTP OpenAI-Compatible Web2API Proxy for Xiaomi MiMo Studio.
Zero browser downloads, zero headless Chrome running in background during proxy serve.
Ultra-lightweight, sub-second TTFT, native SSE streaming with exact token usage.
Compatible with OpenRouter, 9router, Hermes, OpenCode, Claude Code, and Cherry Studio.
"""

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
    import httpx
    import uvicorn
except ImportError:
    print("[mimoly] Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from json_repair import loads as json_repair_loads
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    json_repair_loads = None


def safe_json_loads(text: str, default=None):
    """Robust JSON loader: tries json_repair first, falls back to stdlib."""
    if default is None:
        default = {}
    if JSON_REPAIR_AVAILABLE:
        try:
            return json_repair_loads(text)
        except Exception:
            pass
    try:
        decoder = json.JSONDecoder()
        idx = text.find("{") if "{" in text else text.find("[")
        if idx == -1:
            return default
        obj, _ = decoder.raw_decode(text, idx)
        return obj
    except Exception:
        return default


def normalize_upstream_frame(parsed: Any) -> Optional[Dict[str, Any]]:
    """Return the upstream SSE frame as a dict, or None when it carries no
    chat-completion data.

    Xiaomi's upstream mixes frame shapes on the same stream: normal content
    frames (``{"type":"text","content":...}``), usage frames (``{"promptTokens":...}``),
    and internal web-search payloads that arrive as a JSON *array*
    (``[{"id":..,"text":"# Historical weather .."}]``). The latter must be ignored
    instead of crashing the reader with ``'list' object has no attribute 'get'``.
    """
    return parsed if isinstance(parsed, dict) else None


# Xiaomi MiMo Studio performs its own web retrieval for any prompt that contains
# a literal URL and hands the result to the model as pre-fetched context. The
# model then reads "the page is already available" and answers from that payload
# instead of calling the client's browser tool -- so a user who pastes a link
# gets a summary and no tool call, even though the toolset is loaded.
#
# The retrieval is triggered server-side on a recognisable URL, so we break the
# literal form before sending the prompt upstream. Splitting on "." and "/" with
# surrounding spaces leaves the URL perfectly readable to the model (which
# reassembles it, and we measured it emits the CLEAN url when it calls the
# tool), while the upstream injector no longer sees a URL to fetch.
# Measured: raw URL 0/16 tool calls vs masked 15/16 across four page shapes.
_URL_RE = re.compile(r'https?://[^\s<>"\')]+')


def mask_urls_for_upstream(text: str) -> str:
    """Defuse Xiaomi's server-side webSearch injector by spacing out URLs.

    ``https://en.wikipedia.org/wiki/X`` -> ``https://en . wikipedia . org / wiki / X``
    The scheme stays intact so the model still recognises it as a URL.
    """
    if not text or "http" not in text:
        return text

    def _mask(match: "re.Match[str]") -> str:
        url = match.group(0)
        m = re.match(r"(https?://)(.*)$", url, re.DOTALL)
        if not m:
            return url
        scheme, rest = m.group(1), m.group(2)
        # Trailing punctuation belongs to the sentence, not the URL.
        trail = ""
        while rest and rest[-1] in ".,;:!?":
            trail = rest[-1] + trail
            rest = rest[:-1]
        rest = rest.replace(".", " . ").replace("/", " / ")
        return f"{scheme}{rest}{trail}"

    return _URL_RE.sub(_mask, text)


def sanitize_observation(output: str, max_chars: int = 2000) -> str:
    """Truncate tool observations head/tail to keep context lean and prevent refusal."""
    if not output or len(output) <= max_chars:
        return output
    half = max_chars // 2
    return (
        f"{output[:half]}\n\n"
        f"[... {len(output) - max_chars} characters truncated by mimoly ...]\n\n"
        f"{output[-half:]}"
    )

BASE_DIR = Path(__file__).parent.resolve()
SESSION_FILE = BASE_DIR / "session.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
CHAT_API_URL = "https://aistudio.xiaomimimo.com/open-apis/bot/chat"
CHAT_CONV_SAVE_URL = "https://aistudio.xiaomimimo.com/open-apis/chat/conversation/save"

MODEL_CATALOG = [
    {
        "id": "mimo-v2.6-pro",
        "name": "MiMo-V2.6-Pro (Flagship Deep Reasoning)",
        "upstream_model": "mimo-v2.6-pro",
        "description": "Xiaomi flagship reasoning model with maximum deep thinking and complex problem solving."
    },
    {
        "id": "mimo-v2.6-flash",
        "name": "MiMo-V2.6-Flash (High-Speed & Smart)",
        "upstream_model": "mimo-v2.6-flash",
        "description": "Xiaomi high-speed, responsive model with full reasoning capabilities."
    },
]

MODEL_ALIASES = {
    # Pro / Flagship
    "mimo-v2.6-pro": "mimo-v2.6-pro",
    "mimo-v2.5-pro": "mimo-v2.6-pro",
    "mimo-v2-pro": "mimo-v2.6-pro",
    "mimo-v2.1-pro": "mimo-v2.6-pro",
    "mimo-pro": "mimo-v2.6-pro",
    "mimo": "mimo-v2.6-pro",
    # Flash / Fast
    "mimo-v2.6-flash": "mimo-v2.6-flash",
    "mimo-v2.5": "mimo-v2.6-flash",
    "mimo-v2-flash": "mimo-v2.6-flash",
    "mimo-v2.1-omni": "mimo-v2.6-flash",
    "mimo-flash": "mimo-v2.6-flash",
}

app = FastAPI(title="Mimoly Web2API Proxy", description="100% Pure HTTP OpenAI-Compatible Proxy for Xiaomi MiMo")

# Allow CORS for web frontends (Cherry Studio, NextChat, OpenRouter, 9router, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stats (reset on restart)
# ---------------------------------------------------------------------------
_START_TIME = time.time()
_STATS = {
    "requests": 0,
    "errors": 0,
    "tool_calls": 0,
    "coerced_args": 0,
    "invalid_args": 0,        # args still mismatched after coercion
    "prompt_tokens": 0,       # cumulative input tokens
    "completion_tokens": 0,   # cumulative output tokens
    "total_tokens": 0,
    "reasoning_tokens": 0,
    "tool_usage": {},        # {tool_name: count}
    "model_usage": {},       # {model_id: count}
    "model_tokens": {},      # {model_id: {prompt_tokens, completion_tokens, total_tokens}}
}


def _record_usage(usage: Dict[str, Any], model: str = "") -> None:
    """Accumulate upstream token usage into the global + per-model counters."""
    pt = usage.get("prompt_tokens", 0) or 0
    ct = usage.get("completion_tokens", 0) or 0
    tt = usage.get("total_tokens", 0) or 0
    rt = (usage.get("completion_tokens_details", {}) or {}).get("reasoning_tokens", 0) or 0
    _STATS["prompt_tokens"] += pt
    _STATS["completion_tokens"] += ct
    _STATS["total_tokens"] += tt
    _STATS["reasoning_tokens"] += rt
    if model:
        mt = _STATS["model_tokens"].setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        mt["prompt_tokens"] += pt
        mt["completion_tokens"] += ct
        mt["total_tokens"] += tt


def get_session_cookies() -> Dict[str, str]:
    """Load and sanitize cookies from session.json, stripping surrounding double-quotes."""
    if not SESSION_FILE.exists():
        raise FileNotFoundError(
            f"Session file not found at {SESSION_FILE}. "
            "Please run 'python mimoly.py login' once to authenticate."
        )
    with open(SESSION_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cookies = {}
    if isinstance(raw, list):
        for c in raw:
            name = c.get("name")
            val = str(c.get("value", "")).strip('"')
            if name:
                cookies[name] = val
    elif isinstance(raw, dict):
        for k, v in raw.items():
            cookies[k] = str(v).strip('"')

    required = ["userId", "xiaomichatbot_ph", "xiaomichatbot_serviceToken"]
    missing = [r for r in required if r not in cookies]
    if missing:
        print(f"[mimoly] Warning: Missing recommended auth cookies: {missing}")

    return cookies


AGENT_PROFILES = {
    "default": {
        "label": "Default (Qwen XML native)",
        "tool_intro": (
            "Kamu adalah asisten AI dengan kemampuan tool calling. Gunakan format Qwen XML native.\n"
            "Berikut tools yang tersedia:\n{tool_desc}\n\n"
            "Cara memanggil tool (pakai SATU format ini, jangan format lain):\n"
            "<tool_call>\n"
            "<function=nama_tool>\n"
            "<parameter=param1>nilai</parameter>\n"
            "<parameter=param2>nilai</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "Atau jika tool butuh argumen kompleks:\n"
            "<tool_call>\n"
            "<function=nama_tool>\n"
            '{\n  "param1": "nilai",\n  "param2": "nilai"\n}\n'
            "</function>\n"
            "</tool_call>\n\n"
            "PENTING:\n"
            "- Panggil SATU tool per blok <tool_call>.\n"
            "- Selalu tutup dengan </function></tool_call>.\n"
            "- Jangan ulangi tool call yang sama tanpa membaca hasil observasi dulu.\n"
            "- Jika menulis kode program, tuliskan kode lengkap fungsional sekarang juga."
        ),
    },
    "claude-code": {
        "label": "Claude Code (Anthropic XML/JSON hybrid)",
        "tool_intro": (
            "Kamu adalah AI coding agent. Gunakan format tool call ala Claude Code (Anthropic style):\n"
            "Untuk memanggil tool, emit JSON tool_use block:\n"
            "```json\n"
            '{"type": "tool_use", "id": "toolu_<random>", "name": "<nama_tool>", "input": {<params>}}\n'
            "```\n\n"
            "Atau gunakan format XML <tool_call>...</tool_call> dengan <function=nama_tool> dan <parameter=kunci>nilai</parameter>.\n\n"
            "Tools yang tersedia:\n{tool_desc}\n\n"
            "ATURAN:\n"
            "- Panggil SATU tool per turn, inspect hasilnya, baru panggil berikutnya.\n"
            "- Untuk code generation: tulis kode lengkap di markdown ``` blok bahasa.\n"
            "- Jangan haluskan jawaban dengan teks bertele-tele, langsung kerjakan."
        ),
    },
    "codex": {
        "label": "OpenAI Codex (strict JSON function calling)",
        "tool_intro": (
            "You are an AI coding assistant. Use OpenAI strict JSON function calling format.\n"
            "Available tools:\n{tool_desc}\n\n"
            "When you need a tool, emit EXACTLY this JSON shape and nothing else in the same turn:\n"
            "```json\n"
            '{"tool_calls": [{"id": "call_<random>", "type": "function", "function": {"name": "<nama_tool>", "arguments": "<JSON-stringified params>"}}]}\n'
            "```\n\n"
            "RULES:\n"
            "- Arguments MUST be a JSON string (escaped), not an object.\n"
            "- One tool call per turn unless parallel calls are independent.\n"
            "- Read the tool result, then decide the next step.\n"
            "- If the task is a code task, output the full working code immediately."
        ),
    },
    "opencode": {
        "label": "OpenCode (XML/JSON tolerant)",
        "tool_intro": (
            "Kamu adalah AI coding agent. Pakai format OpenCode yang fleksibel (XML atau JSON).\n"
            "Tools tersedia:\n{tool_desc}\n\n"
            "Cara panggil tool (pilih salah satu, konsisten dalam satu turn):\n\n"
            "FORMAT 1 - XML (Qwen-style):\n"
            "<tool_call>\n"
            "<function=nama_tool>\n"
            "<parameter=kunci>nilai</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "FORMAT 2 - JSON markdown:\n"
            "```json\n"
            '{"tool_calls": [{"name": "nama_tool", "arguments": {"kunci": "nilai"}}]}\n'
            "```\n\n"
            "Pilih satu format dan pakai konsisten. Jangan campur dalam satu turn."
        ),
    },
    "hermes": {
        "label": "Hermes CLI (Qwen XML strict)",
        "tool_intro": (
            "Kamu adalah AI agent di Hermes CLI. SELALU gunakan format Qwen XML ini untuk tool call:\n"
            "<tool_call>\n"
            "<function=nama_tool>\n"
            "<parameter=kunci>nilai</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "Tools tersedia:\n{tool_desc}\n\n"
            "ATURAN KERAS:\n"
            "- SELALU tutup dengan </function></tool_call>.\n"
            "- Panggil SATU tool per blok, inspect hasilnya, baru lanjut.\n"
            "- Path file: SELALU gunakan path relatif terhadap root project yang persis sama, contoh: 'src/app/page.tsx'.\n"
            "- Jika user minta tulis kode, langsung tulis kode lengkap fungsional sekarang juga."
        ),
    },
    "pi": {
        "label": "Pi / Oh My Pi (plain text + simple XML)",
        "tool_intro": (
            "Kamu asisten coding. Untuk panggil tool, gunakan format XML sederhana:\n"
            "<tool_call>\n"
            "<function=nama_tool>\n"
            "<parameter=kunci>nilai</parameter>\n"
            "</function>\n"
            "</tool_call>\n\n"
            "Tools:\n{tool_desc}\n\n"
            "Prinsip: ringkas, satu tool per turn, langsung tulis kode kalau diminta."
        ),
    },
}


def detect_agent_framework(request: Request, body: dict) -> str:
    """Detect which agent framework is calling, from header > body > env."""
    framework = request.headers.get("X-Agent-Framework", "").lower().strip()
    if framework in AGENT_PROFILES:
        return framework
    framework = (body.get("agent_framework") or body.get("framework") or "").lower().strip()
    if framework in AGENT_PROFILES:
        return framework
    env_framework = os.environ.get("MIMOLY_FRAMEWORK", "").lower().strip()
    if env_framework in AGENT_PROFILES:
        return env_framework
    return "default"


def build_agent_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, framework: str = "default") -> str:
    """Format dialogue history and tool definitions cleanly for MiMo."""
    if not tools:
        formatted = []
        for m in messages:
            r = m.get("role", "user")
            c = m.get("content", "")
            if isinstance(c, list):
                c = " ".join([part.get("text", "") for part in c if isinstance(part, dict) and "text" in part])
            clean_c = re.sub(r"<EXTREMELY_IMPORTANT>[\s\S]*?</EXTREMELY_IMPORTANT>", "", c).strip()
            clean_c = re.sub(r"<SUBAGENT-STOP>[\s\S]*?$", "", clean_c).strip()
            if r == "system":
                formatted.append(f"Instruksi: {clean_c}")
            elif r == "user":
                formatted.append(f"User: {clean_c}")
            elif r == "assistant":
                formatted.append(f"Assistant: {clean_c}")
        return "\n\n".join(formatted)

    history = []
    user_goals = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            content = " ".join(text_parts)

        if role == "system":
            if content and len(content) < 1500:
                history.append(f"[System]: {content}")
        elif role == "user":
            # Bersihkan bootstrap tags agar prompt tetap ringkas & fokus
            clean_c = re.sub(r"<EXTREMELY_IMPORTANT>[\s\S]*?</EXTREMELY_IMPORTANT>", "", content).strip()
            clean_c = re.sub(r"<SUBAGENT-STOP>[\s\S]*?$", "", clean_c).strip()
            clean_c = clean_c if clean_c else content
            # Defuse Xiaomi's server-side webSearch injector: a literal URL in the
            # prompt makes upstream pre-fetch the page and answer from it instead
            # of letting the model call the browser tool. See mask_urls_for_upstream.
            clean_c = mask_urls_for_upstream(clean_c)
            user_goals.append(clean_c)
            history.append(f"[User]: {clean_c}")
        elif role == "assistant":
            tool_calls = m.get("tool_calls", [])
            if tool_calls:
                tc_names = [tc.get("function", {}).get("name", "") for tc in tool_calls]
                history.append(f"[Assistant Action]: Menjalankan tool: {', '.join(tc_names)}")
            elif content:
                history.append(f"[Assistant]: {content}")
        elif role == "tool":
            tool_name = m.get("name", "tool")
            sanitized = sanitize_observation(content, max_chars=2000)
            # Tool results routinely contain JSON, which is full of double quotes.
            # Embedding that verbatim into this plain-text prompt leaves an
            # unescaped `{"` sequence that the model cannot parse, so it retries
            # the same tool until it burns its turn budget. Neutralise the quotes
            # so the observation reads as plain text.
            sanitized = sanitized.replace('"', "'")
            history.append(f"[Hasil {tool_name}]: {sanitized}")

    main_goal = user_goals[-1] if user_goals else ""

    prompt_lines = []
    if tools:
        tool_desc = []
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name")
            desc = (fn.get("description") or "").split("\n")[0].split(". ")[0].strip()
            params = list(fn.get("parameters", {}).get("properties", {}).keys())
            param_str = ", ".join(params)
            tool_desc.append(f"- {name}({param_str}): {desc}")

        profile = AGENT_PROFILES.get(framework, AGENT_PROFILES["default"])
        intro_text = profile["tool_intro"].replace("{tool_desc}", "\n".join(tool_desc))
        prompt_lines.append(intro_text)

    if history:
        # Keep recent history
        recent_hist = history[-8:]
        prompt_lines.append("Riwayat percakapan:\n" + "\n".join(recent_hist))

    if main_goal:
        if messages and messages[-1].get("role") == "tool":
            prompt_lines.append(
                f"Tugas utama user: {main_goal}\n"
                f"Status terkini: Hasil eksekusi tool terbaru ada di riwayat di atas.\n"
                f"- Jika tugas utama sudah terjawab/selesai secara lengkap, berikan jawaban akhir yang jelas dan informatif kepada user sekarang.\n"
                f"- Jika tugas masih membutuhkan langkah berikutnya atau tool lanjutan (misal melihat isi skill, memanggil MCP, membaca file, delegasi task), panggil tool berikutnya sekarang."
            )
        else:
            prompt_lines.append(f"Tugas sekarang:\n{main_goal}")

    return "\n\n".join(prompt_lines)


def _coerce_to_list(val: Any) -> List[Any]:
    """Best-effort turn a stringified array into a real list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            parsed = safe_json_loads(s, default=None)
            if isinstance(parsed, list):
                return parsed
            inner = s[1:-1].strip()
            if not inner:
                return []
            # Fallback: comma-separated items, strip surrounding quotes.
            return [p.strip().strip("\"'") for p in inner.split(",") if p.strip()]
        return [s] if s else []
    return [val]


def _schema_for(name: str, available_tools: Optional[List[Dict[str, Any]]]) -> Optional[dict]:
    """Return the declared parameter schema for a tool, or None."""
    if not available_tools:
        return None
    for t in available_tools:
        fn = t.get("function") or {}
        if fn.get("name") == name:
            schema = fn.get("parameters") or {}
            return schema if isinstance(schema, dict) else None
    return None


def _type_list(spec: dict) -> List[str]:
    """Normalize a JSON-schema ``type`` (string or union list) to a list."""
    want = spec.get("type")
    if isinstance(want, list):
        return [w for w in want if isinstance(w, str)]
    return [want] if isinstance(want, str) else []


def _coerce_value(val: Any, spec: dict) -> Any:
    """Recursively coerce ``val`` toward the type declared by ``spec``."""
    if not isinstance(spec, dict):
        return val
    types = _type_list(spec)

    # Nullable: JSON string "null" -> Python None
    if isinstance(val, str) and val.strip().lower() == "null" and "null" in types:
        return None

    target = next((t for t in types if t and t != "null"), None)

    if target == "array" and not isinstance(val, list):
        val = _coerce_to_list(val)
    elif target == "object":
        if isinstance(val, str):
            parsed = safe_json_loads(val.strip(), default=None)
            if isinstance(parsed, dict):
                val = parsed
        if isinstance(val, dict):
            for k, sub in (spec.get("properties") or {}).items():
                if k in val:
                    val[k] = _coerce_value(val[k], sub)
    elif target == "integer" and isinstance(val, str) and val.strip().lstrip("-").isdigit():
        val = int(val.strip())
    elif target == "number" and isinstance(val, str):
        try:
            val = float(val.strip())
        except ValueError:
            pass
    elif target == "boolean":
        if isinstance(val, str) and val.strip().lower() in ("true", "false", "1", "0"):
            val = val.strip().lower() in ("true", "1")
        elif isinstance(val, int) and not isinstance(val, bool) and val in (0, 1):
            val = bool(val)

    # Array-of-objects: repair item key names + recurse into item fields.
    # e.g. Hermes' tool_call wants calls=[{name, arguments}]; MiMo emits
    # calls=[{tool, arguments}], and the call is rejected for a missing 'name'.
    if target == "array" and isinstance(val, list):
        item_spec = spec.get("items")
        if isinstance(item_spec, dict) and item_spec.get("type") == "object":
            item_props = item_spec.get("properties") or {}
            aliases = {"name": ("tool", "tool_name", "function", "function_name", "id")}
            for element in val:
                if not isinstance(element, dict):
                    continue
                for want_key, alt_keys in aliases.items():
                    if want_key in item_props and want_key not in element:
                        for alt in alt_keys:
                            if alt in element:
                                element[want_key] = element.pop(alt)
                                break
                for k, sub in item_props.items():
                    if k in element:
                        element[k] = _coerce_value(element[k], sub)
    return val


def _value_matches(val: Any, types: List[str]) -> bool:
    """True if ``val`` satisfies at least one of the declared JSON types."""
    if not types:
        return True
    for t in types:
        if t == "null" and val is None:
            return True
        if t == "string" and isinstance(val, str):
            return True
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return True
        if t == "number" and isinstance(val, (int, float)) and not isinstance(val, bool):
            return True
        if t == "boolean" and isinstance(val, bool):
            return True
        if t == "array" and isinstance(val, list):
            return True
        if t == "object" and isinstance(val, dict):
            return True
    return False


def validate_tool_args(name: str, args: dict,
                       available_tools: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Check ``args`` against the declared tool schema.

    Returns ``{"valid": bool, "issues": [str, ...]}``. Used after coercion to
    surface (not silently swallow) arguments that still don't fit the schema.
    """
    if not isinstance(args, dict):
        return {"valid": False, "issues": ["arguments is not a JSON object"]}
    schema = _schema_for(name, available_tools)
    if not isinstance(schema, dict):
        return {"valid": True, "issues": []}

    props = schema.get("properties") or {}
    required = schema.get("required") or []
    issues: List[str] = []

    for req in required:
        if req not in args:
            issues.append(f"missing required field '{req}'")
    for key, spec in props.items():
        if key not in args or not isinstance(spec, dict):
            continue
        types = _type_list(spec)
        if types and not _value_matches(args[key], types):
            issues.append(
                f"field '{key}' expected {'|'.join(types)}, got {type(args[key]).__name__}"
            )
    return {"valid": not issues, "issues": issues}


def coerce_tool_args(name: str, args: dict, available_tools: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Repair tool arguments whose JSON type drifted from the declared schema.

    MiMo (the upstream model) sometimes emits a parameter that *should* be an
    array/object/number/boolean as a JSON *string* — e.g. ``queries='["a","b"]'``
    instead of ``queries=["a","b"]``. Hermes' ``tool_search``/``tool_describe``
    then reject the call (``requires a 'name'`` / ``not_found``), the model gives
    up on the tool, and the MCP toolset is never actually used. We coerce the
    value back to the type the schema declares (recursively) so the call survives.
    """
    if not isinstance(args, dict) or not available_tools:
        return args

    snapshot = repr(args)  # cheap before-snapshot for coercion detection
    schema = _schema_for(name, available_tools)
    if not isinstance(schema, dict):
        return args

    for key, spec in (schema.get("properties") or {}).items():
        if key in args and isinstance(spec, dict):
            args[key] = _coerce_value(args[key], spec)

    if repr(args) != snapshot:
        _STATS["coerced_args"] += 1
    return args


def normalize_tool_args(name: str, args: dict, user_prompt: str = "",
                        available_tools: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Normalize tool arguments for consistency across frameworks and models."""
    if not isinstance(args, dict):
        return args

    # Alias normalization
    if "file" in args and "path" not in args:
        args["path"] = args.pop("file")
    if "filename" in args and "path" not in args:
        args["path"] = args.pop("filename")
    if "filepath" in args and "path" not in args:
        args["path"] = args.pop("filepath")
    if "cmd" in args and "command" not in args:
        args["command"] = args.pop("cmd")
    if name == "search_files" and "query" in args and "pattern" not in args:
        args["pattern"] = args.pop("query")
    if name == "patch":
        if "old_code" in args and "old_string" not in args:
            args["old_string"] = args.pop("old_code")
        if "new_code" in args and "new_string" not in args:
            args["new_string"] = args.pop("new_code")
        if "find" in args and "old_string" not in args:
            args["old_string"] = args.pop("find")
        if "replace" in args and "new_string" not in args:
            args["new_string"] = args.pop("replace")

    # Remove pagination parameters from read_file to avoid Hermes CLI 'stale_write_blocked (partial view)' refusal
    if name == "read_file":
        args.pop("limit", None)
        args.pop("offset", None)

    # Infer missing path from user_prompt if write_file/read_file omitted it
    if name in ["write_file", "read_file", "patch"] and "path" not in args and user_prompt:
        m_p = re.search(r"['\"]([a-zA-Z0-9_\-/\\]+\.[a-zA-Z0-9]+)['\"]", user_prompt)
        if m_p:
            args["path"] = m_p.group(1)

    # Path normalization for file tools
    if "path" in args and isinstance(args["path"], str):
        p = args["path"].replace("\\", "/").strip()
        if p.startswith("D:/Software/Hermes Workspace/"):
            p = p[len("D:/Software/Hermes Workspace/"):]
        if "spectra" in user_prompt.lower():
            if p in ["page.tsx", "layout.tsx", "globals.css"]:
                p = f"projects/websites/spectra/src/app/{p}"
            elif p.startswith("src/app/") or p.startswith("src/"):
                p = f"projects/websites/spectra/{p}"
        else:
            m_target = re.search(r"(projects/[a-zA-Z0-9_\-]+(?:/[a-zA-Z0-9_\-]+)?)", user_prompt)
            if m_target:
                target_base = m_target.group(1).rstrip("/")
                if not p.startswith("projects/") and not p.startswith("/") and not (len(p) > 2 and p[1] == ":"):
                    p = f"{target_base}/{p}"
        # Ensure path under projects/ is absolute to prevent terminal CWD double-prefixing
        if p.startswith("projects/"):
            p = f"D:/Software/Hermes Workspace/{p}"
        args["path"] = p

    # Repair JSON-type drift (stringified arrays/objects) against the tool schema.
    args = coerce_tool_args(name, args, available_tools)

    # Surface (don't swallow) args that still don't fit the schema.
    _validation = validate_tool_args(name, args, available_tools)
    if not _validation["valid"]:
        _STATS["invalid_args"] += 1
        print(f"[mimoly] schema mismatch for '{name}': {_validation['issues']}")

    return args


def smart_extract_tool_calls(reply_text: str, user_prompt: str, available_tools: Optional[List[Dict[str, Any]]] = None):
    """Detect explicit JSON tool_calls or extract code blocks mapped to write/shell tools."""
    tool_names = [t.get("function", {}).get("name") for t in available_tools] if available_tools else []

    # 1. Cek explicit JSON or XML tool_calls
    # Pola 1A: XML-style <function=name> / <invoke name="name"> / <function name="name">
    xml_matches = re.findall(
        r"<(?:function|invoke)(?:=|\s+name=)[\"']?([a-zA-Z0-9_\-]+)[\"']?>([\s\S]*?)</(?:function|invoke)>",
        reply_text,
    )
    if xml_matches:
        tc_list = []
        for fn_name, params_str in xml_matches:
            params = {}
            clean_p = params_str.strip()
            idx = clean_p.find("{")
            if idx != -1:
                try:
                    obj, _ = json.JSONDecoder().raw_decode(clean_p, idx)
                    if isinstance(obj, dict):
                        params = obj
                except Exception:
                    pass
            if not params:
                param_matches = re.findall(
                    r"<(?:parameter|param)(?:=|\s+name=)[\"']?([a-zA-Z0-9_\-]+)[\"']?>([\s\S]*?)</(?:parameter|param)>",
                    params_str,
                )
                for p_name, p_val in param_matches:
                    p_val = p_val.strip()
                    if p_val.isdigit():
                        params[p_name] = int(p_val)
                    elif p_val.lower() == "true":
                        params[p_name] = True
                    elif p_val.lower() == "false":
                        params[p_name] = False
                    else:
                        params[p_name] = p_val

            params = normalize_tool_args(fn_name, params, user_prompt, available_tools)
            if fn_name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                fn_name = "terminal"
            tc_list.append({"name": fn_name, "arguments": params})
        if tc_list:
            clean = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", reply_text).strip()
            clean = re.sub(r"<(?:function|invoke)[\s\S]*?</(?:function|invoke)>", "", clean).strip()
            clean = clean.replace("</think>", "").strip()
            return tc_list, clean

    # Pola 1B: <tool_call> ... </tool_call>
    m_tc = re.findall(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", reply_text)
    if m_tc:
        tc_list = []
        for block in m_tc:
            clean_block = block.replace("</think>", "").strip()
            # Try safe_json_loads first (json-repair is more tolerant)
            tc_obj = safe_json_loads(clean_block, default=None)
            if isinstance(tc_obj, list):
                for item in tc_obj:
                    if isinstance(item, dict):
                        if "tool_calls" in item and isinstance(item["tool_calls"], list):
                            for tc in item["tool_calls"]:
                                name = tc.get("name")
                                args = normalize_tool_args(name, tc.get("arguments") or tc.get("parameters") or {}, user_prompt, available_tools)
                                if name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                                    name = "terminal"
                                tc_list.append({"name": name, "arguments": args})
                        elif "name" in item:
                            name = item["name"]
                            args = normalize_tool_args(name, item.get("arguments") or item.get("parameters") or {}, user_prompt, available_tools)
                            if name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                                name = "terminal"
                            tc_list.append({"name": name, "arguments": args})
            elif isinstance(tc_obj, dict):
                if "tool_calls" in tc_obj and isinstance(tc_obj["tool_calls"], list):
                    for tc in tc_obj["tool_calls"]:
                        name = tc.get("name")
                        args = normalize_tool_args(name, tc.get("arguments") or tc.get("parameters") or {}, user_prompt, available_tools)
                        if name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                            name = "terminal"
                        tc_list.append({"name": name, "arguments": args})
                elif "name" in tc_obj:
                    name = tc_obj["name"]
                    args = normalize_tool_args(name, tc_obj.get("arguments") or tc_obj.get("parameters") or {}, user_prompt, available_tools)
                    if name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                        name = "terminal"
                    tc_list.append({"name": name, "arguments": args})
        if tc_list:
            clean = re.sub(r"<tool_call>[\s\S]*?(?:</tool_call>|$)", "", reply_text).strip()
            clean = clean.replace("</think>", "").strip()
            return tc_list, clean

    m_json = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", reply_text)
    if m_json:
        try:
            d = json.loads(m_json.group(1), strict=False)
            if "tool_calls" in d and isinstance(d["tool_calls"], list):
                clean = reply_text.replace(m_json.group(0), "").strip()
                calls = []
                for tc in d["tool_calls"]:
                    nm = tc.get("name") or (tc.get("function") or {}).get("name")
                    ar = tc.get("arguments") or tc.get("parameters") or (tc.get("function") or {}).get("arguments") or {}
                    if isinstance(ar, str):
                        ar = safe_json_loads(ar, default={})
                    calls.append({"name": nm, "arguments": normalize_tool_args(nm, ar, user_prompt, available_tools)})
                return calls, clean
            elif "name" in d and ("parameters" in d or "arguments" in d):
                args = d.get("parameters") or d.get("arguments") or {}
                clean = reply_text.replace(m_json.group(0), "").strip()
                return [{"name": d["name"], "arguments": normalize_tool_args(d["name"], args, user_prompt, available_tools)}], clean
        except Exception:
            pass

    m2 = re.search(r"\{\s*\"tool_calls\"\s*:\s*(\[[\s\S]*?\])\s*\}", reply_text)
    if m2:
        try:
            arr = json.loads(m2.group(1), strict=False)
            calls = []
            for item in arr:
                if "name" in item:
                    args = item.get("arguments") or item.get("parameters") or {}
                    if isinstance(args, str):
                        args = safe_json_loads(args, default={})
                    calls.append({"name": item.get("name"),
                                  "arguments": normalize_tool_args(item.get("name"), args, user_prompt, available_tools)})
            if calls:
                clean = reply_text.replace(m2.group(0), "").strip()
                return calls, clean
        except Exception:
            pass

    calls = []
    # 2. Extract shell/bash commands if shell tool is available
    if "shell" in tool_names or "bash" in tool_names:
        shell_tool = "shell" if "shell" in tool_names else "bash"
        shell_blocks = list(re.finditer(r"```(?:bash|sh|cmd|powershell)\n([\s\S]*?)\n```", reply_text))
        for sb in shell_blocks:
            cmd = sb.group(1).strip()
            if cmd and not cmd.startswith("#"):
                calls.append({
                    "name": shell_tool,
                    "arguments": {
                        "command": cmd
                    }
                })

    # 3. Extract code blocks with file names if write tool is available
    if "write" in tool_names:
        code_blocks = list(re.finditer(r"```([a-zA-Z0-9_\-\.]+)?\n([\s\S]*?)\n```", reply_text))
        prompt_filenames = re.findall(r"\b([a-zA-Z0-9_\-/\\]+\.[a-zA-Z0-9]+)\b", user_prompt)
        used_filenames = set()

        for idx, cb in enumerate(code_blocks):
            lang = (cb.group(1) or "").lower()
            code = cb.group(2)
            if lang in ["text", "bash", "sh", "output", "cmd", "powershell"]:
                continue

            if code.strip().startswith("python\n"):
                code = code.strip()[7:].lstrip()

            filename = None
            first_line = code.strip().split("\n")[0] if code.strip() else ""
            m_comment = re.search(r"^(?:#|//|<!--)\s*([a-zA-Z0-9_\-/\\]+\.[a-zA-Z0-9]+)", first_line)
            if m_comment:
                candidate = m_comment.group(1).replace("\\", "/")
                valid_exts = ('.js', '.mjs', '.ts', '.tsx', '.json', '.css', '.html', '.md', '.py', '.sh', '.bat', '.env', '.yaml', '.yml')
                if candidate.endswith(valid_exts):
                    filename = candidate

            if not filename:
                start_pos = cb.start()
                preceding_text = reply_text[max(0, start_pos - 150):start_pos]
                m_prec = re.findall(r"\b([a-zA-Z0-9_\-/\\]+\.(?:js|mjs|ts|tsx|json|css|html|md|py|sh|bat))\b", preceding_text)
                if m_prec:
                    filename = m_prec[-1].replace("\\", "/")

            if filename and filename.endswith(".json"):
                code = re.sub(r"^\s*//[^\n]*\n", "", code).strip()

            if not filename:
                for pf in prompt_filenames:
                    clean_pf = pf.replace("\\", "/")
                    if clean_pf not in used_filenames:
                        filename = clean_pf
                        break

            if filename:
                used_filenames.add(filename)
                calls.append({
                    "name": "write",
                    "arguments": {
                        "path": filename,
                        "content": code
                    }
                })

    return calls, reply_text


@app.get("/")
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "mimoly-web2api",
        "version": "3.0.0",
        "auth_configured": SESSION_FILE.exists()
    }


@app.get("/v1/stats")
async def stats():
    top_tools = sorted(_STATS["tool_usage"].items(), key=lambda x: -x[1])
    return {
        "uptime_seconds": round(time.time() - _START_TIME, 1),
        "requests": _STATS["requests"],
        "errors": _STATS["errors"],
        "tool_calls": _STATS["tool_calls"],
        "coerced_args": _STATS["coerced_args"],
        "invalid_args": _STATS["invalid_args"],
        "tokens": {
            "prompt_tokens": _STATS["prompt_tokens"],
            "completion_tokens": _STATS["completion_tokens"],
            "total_tokens": _STATS["total_tokens"],
            "reasoning_tokens": _STATS["reasoning_tokens"],
        },
        "tool_usage": dict(top_tools[:20]),
        "model_usage": _STATS["model_usage"],
        "model_tokens": _STATS["model_tokens"],
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Ultra-light single-file live dashboard for /v1/stats (no deps, auto-refresh)."""
    return HTMLResponse(_DASHBOARD_HTML)


_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mimoly dashboard</title>
<style>
  :root {
    --bg: #0a0e14; --surface: #12171e; --card: #161c24; --bd: #252d38;
    --fg: #d1d9e6; --fg2: #e8edf4; --mut: #6b7a8d;
    --acc: #4d9eff; --grn: #34d058; --red: #ea4a5a; --yel: #e3b341;
    --radius: 12px;
  }
  * { box-sizing: border-box; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--fg); min-height: 100vh;
  }
  .wrap { max-width: 960px; margin: 0 auto; padding: 28px 20px 48px; }

  /* ── Header ── */
  header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
  header h1 { font-size: 20px; font-weight: 700; color: var(--fg2); letter-spacing: -.3px; }
  .badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: .4px;
    padding: 3px 10px; border-radius: 20px;
    background: rgba(52,208,88,.12); color: var(--grn);
    transition: all .3s;
  }
  .badge.off { background: rgba(234,74,90,.12); color: var(--red); }
  .badge i {
    width: 6px; height: 6px; border-radius: 50%; background: currentColor;
  }
  .badge.on i { animation: pulse 2s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
  .meta { color: var(--mut); font-size: 12px; margin-bottom: 24px; }

  /* ── Section groups ── */
  .section { margin-bottom: 20px; }
  .section-title {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .8px; color: var(--mut); margin-bottom: 8px;
    padding-left: 2px;
  }

  /* ── Card grid ── */
  .g3 { display: grid; gap: 10px; grid-template-columns: repeat(3, 1fr); }
  .g4 { display: grid; gap: 10px; grid-template-columns: repeat(4, 1fr); }
  .g2 { display: grid; gap: 10px; grid-template-columns: repeat(2, 1fr); }
  @media (max-width: 640px) { .g3,.g4 { grid-template-columns: repeat(2,1fr); } }
  @media (max-width: 400px) { .g3,.g4,.g2 { grid-template-columns: 1fr; } }

  .card {
    background: var(--card); border: 1px solid var(--bd);
    border-radius: var(--radius); padding: 16px 18px;
    display: flex; flex-direction: column; gap: 2px;
    transition: border-color .2s;
  }
  .card:hover { border-color: #3a4555; }
  .card .label {
    font-size: 11px; font-weight: 500; text-transform: uppercase;
    letter-spacing: .5px; color: var(--mut);
  }
  .card .val {
    font-size: 28px; font-weight: 700; letter-spacing: -.5px;
    font-family: ui-monospace, 'SF Mono', 'Cascadia Code', 'Consolas', monospace;
    font-variant-numeric: tabular-nums;
    color: var(--fg2); line-height: 1.2;
  }
  .card .unit { font-size: 12px; color: var(--mut); margin-top: 1px; }

  /* Dynamic color: only light up when value > 0 */
  .val[data-color="red"]    { color: var(--red); }
  .val[data-color="yellow"] { color: var(--yel); }
  .val[data-color="green"]  { color: var(--grn); }
  .val[data-color="accent"] { color: var(--acc); }

  /* ── Panels (tables) ── */
  .panel {
    background: var(--card); border: 1px solid var(--bd);
    border-radius: var(--radius); padding: 16px 18px; margin-top: 0;
  }
  .panel table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .panel th {
    text-align: left; font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .6px; color: var(--mut); padding: 0 0 8px;
    border-bottom: 1px solid var(--bd);
  }
  .panel th:last-child { text-align: right; }
  .panel td { padding: 10px 0; border-bottom: 1px solid rgba(37,45,56,.5); }
  .panel td:last-child { text-align: right; font-family: ui-monospace, monospace;
    font-variant-numeric: tabular-nums; font-weight: 600; color: var(--fg2); }
  .panel .bar-wrap {
    height: 4px; background: var(--bd); border-radius: 2px;
    overflow: hidden; margin-top: 4px;
  }
  .panel .bar-fill { height: 100%; background: var(--acc); border-radius: 2px;
    transition: width .4s ease; }
  .panel .empty-state {
    color: var(--mut); font-size: 13px; text-align: center;
    padding: 20px 0;
  }

  /* ── Footer ── */
  .foot {
    margin-top: 32px; padding-top: 16px; border-top: 1px solid var(--bd);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 11px; color: var(--mut);
  }
  .foot a { color: var(--acc); text-decoration: none; }
  .foot a:hover { text-decoration: underline; }
</style>
</head>
<body>
<div class="wrap">

<header>
  <h1>mimoly</h1>
  <span class="badge" id="badge"><i></i><span id="badge-text">connecting</span></span>
</header>
<div class="meta" id="meta">waiting for first poll&hellip;</div>

<!-- ── Traffic ── -->
<div class="section">
  <div class="section-title">Traffic</div>
  <div class="g3">
    <div class="card">
      <span class="label">Requests</span>
      <span class="val" id="req">—</span>
    </div>
    <div class="card">
      <span class="label">Errors</span>
      <span class="val" id="err">—</span>
    </div>
    <div class="card">
      <span class="label">Error rate</span>
      <span class="val" id="erate">—</span>
    </div>
  </div>
</div>

<!-- ── Tool intelligence ── -->
<div class="section">
  <div class="section-title">Tool Intelligence</div>
  <div class="g3">
    <div class="card">
      <span class="label">Tool calls</span>
      <span class="val" id="tc">—</span>
    </div>
    <div class="card">
      <span class="label">Coerced args</span>
      <span class="val" id="co">—</span>
    </div>
    <div class="card">
      <span class="label">Invalid args</span>
      <span class="val" id="iv">—</span>
    </div>
  </div>
</div>

<!-- ── Token usage ── -->
<div class="section">
  <div class="section-title">Token Usage</div>
  <div class="g4">
    <div class="card">
      <span class="label">Input</span>
      <span class="val" id="pt">—</span>
    </div>
    <div class="card">
      <span class="label">Output</span>
      <span class="val" id="ct">—</span>
    </div>
    <div class="card">
      <span class="label">Total</span>
      <span class="val" id="tt">—</span>
    </div>
    <div class="card">
      <span class="label">Reasoning</span>
      <span class="val" id="rt">—</span>
    </div>
  </div>
</div>

<!-- ── Breakdown tables ── -->
<div class="section">
  <div class="section-title">Breakdown</div>
  <div class="g2">
    <div class="panel" id="models"><div class="empty-state">No model data yet</div></div>
    <div class="panel" id="tools"><div class="empty-state">No tool data yet</div></div>
  </div>
</div>

<div class="foot">
  <span>mimoly web2api &middot; <a href="/v1/stats">raw json</a></span>
  <span id="foot-time"></span>
</div>

</div>

<script>
const $ = id => document.getElementById(id);
const fmt = n => n == null ? '—' : n.toLocaleString();

function setVal(id, v, color, suffix) {
  const el = $(id);
  el.textContent = v == null ? '—' : fmt(v) + (suffix || '');
  if (color && v > 0) el.setAttribute('data-color', color);
  else el.removeAttribute('data-color');
}

function buildTable(rows, nameLabel, countLabel) {
  if (!rows.length) return '<div class="empty-state">No data yet</div>';
  const max = Math.max(...rows.map(r => r[1]), 1);
  let h = '<table><tr><th>' + nameLabel + '</th><th>' + countLabel + '</th></tr>';
  for (const [name, val] of rows) {
    const pct = Math.max(3, (val / max) * 100);
    h += '<tr><td>' + name +
      '<div class="bar-wrap"><div class="bar-fill" style="width:' + pct + '%"></div></div>' +
      '</td><td>' + fmt(val) + '</td></tr>';
  }
  return h + '</table>';
}

function uptime(sec) {
  if (!sec) return '0s';
  const d = Math.floor(sec/86400), h = Math.floor(sec%86400/3600),
        m = Math.floor(sec%3600/60), s = Math.floor(sec%60);
  if (d) return d + 'd ' + h + 'h';
  if (h) return h + 'h ' + m + 'm';
  return m + 'm ' + s + 's';
}

let alive = false;
async function tick() {
  try {
    const s = await (await fetch('/v1/stats')).json();
    if (!alive) {
      alive = true;
      $('badge').className = 'badge on';
      $('badge-text').textContent = 'live';
    }

    setVal('req', s.requests, 'accent');
    setVal('err', s.errors, 'red');
    const rate = s.requests > 0 ? ((s.errors / s.requests) * 100) : 0;
    setVal('erate', Math.round(rate * 10) / 10, rate > 5 ? 'red' : rate > 0 ? 'yellow' : null, '%');

    setVal('tc', s.tool_calls);
    setVal('co', s.coerced_args, 'yellow');
    setVal('iv', s.invalid_args, 'red');

    const t = s.tokens || {};
    setVal('pt', t.prompt_tokens);
    setVal('ct', t.completion_tokens, 'green');
    setVal('tt', t.total_tokens, 'accent');
    setVal('rt', t.reasoning_tokens);

    $('meta').textContent = 'up ' + uptime(s.uptime_seconds) +
      ' \u00b7 ' + fmt(s.requests) + ' req \u00b7 ' + fmt(t.total_tokens) + ' tok';

    const mt = s.model_tokens || {};
    const mRows = Object.entries(mt).map(([m, v]) => [m, v.total_tokens]);
    $('models').innerHTML = buildTable(mRows, 'Model', 'Tokens');
    $('tools').innerHTML  = buildTable(Object.entries(s.tool_usage || {}), 'Tool', 'Calls');

    $('foot-time').textContent = new Date().toLocaleTimeString();
  } catch (e) {
    alive = false;
    $('badge').className = 'badge off';
    $('badge-text').textContent = 'offline';
    $('meta').innerHTML = '<span style="color:var(--red)">connection lost</span>';
  }
}
tick();
setInterval(tick, 2000);
</script>
</body>
</html>"""


@app.get("/v1/models")
@app.get("/models")
@app.get("/api/v1/models")
async def list_models():
    now = int(time.time())
    model_list = []
    for m in MODEL_CATALOG:
        model_list.append({
            "id": m["id"],
            "object": "model",
            "created": now,
            "owned_by": "xiaomi",
            "name": m["name"],
            "description": m["description"]
        })
    # Also add custom/ prefix versions for flexible routing
    for m in MODEL_CATALOG:
        model_list.append({
            "id": f"custom/{m['id']}",
            "object": "model",
            "created": now,
            "owned_by": "xiaomi"
        })
    return {"object": "list", "data": model_list}


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    clean_id = model_id.split("/")[-1].lower().strip()
    target_upstream = MODEL_ALIASES.get(clean_id, "mimo-v2.6-pro")
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "xiaomi",
        "upstream_model": target_upstream
    }


@app.get("/api/tags")
async def ollama_tags():
    models = []
    for m in MODEL_CATALOG:
        models.append({
            "name": m["id"],
            "model": m["id"],
            "modified_at": "2026-09-28T00:00:00Z",
            "size": 7000000000,
            "digest": f"sha256:{m['id']}",
            "details": {"format": "gguf", "family": "mimo"}
        })
    return {"models": models}


@app.get("/version")
async def version():
    return {"version": "3.0.0"}


@app.post("/api/show")
async def ollama_show():
    return {
        "license": "Xiaomi MiMo Studio",
        "modelfile": "FROM mimo-v2.5-pro",
        "parameters": "temperature 0.7",
        "template": "{{ .Prompt }}"
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    messages = body.get("messages", [])
    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    stream = body.get("stream", False)
    tools = body.get("tools")
    model = body.get("model", "mimo-v2.6-pro")

    clean_model_id = model.split("/")[-1].lower().strip()
    target_upstream_model = MODEL_ALIASES.get(clean_model_id, "mimo-v2.6-pro")

    # Reasoning effort & Thinking configuration: Natural thinking enabled by default
    reasoning_effort = str(body.get("reasoning_effort", "default")).lower()
    enable_thinking_param = body.get("enable_thinking")
    thinking_param = body.get("thinking")

    thinking_enabled = True
    if reasoning_effort in ["none", "off", "false", "0"]:
        thinking_enabled = False
    elif enable_thinking_param is False:
        thinking_enabled = False
    elif isinstance(thinking_param, dict) and thinking_param.get("type") == "disabled":
        thinking_enabled = False

    # Default to natural thinking (bawaan MiMo 2.6 Pro, fast & focused, no artificial overthinking).
    # Hanya aktifkan amplifier "rata kanan" jika eksplisit diminta via reasoning_effort='max'/'extreme'/'rata_kanan'
    is_rata_kanan = False
    if reasoning_effort in ["max", "high", "extreme", "rata_kanan"] or body.get("rata_kanan") is True:
        is_rata_kanan = True

    print(f"[mimoly] Incoming request: model={model} -> {target_upstream_model}, stream={stream}, messages={len(messages)}, tools={len(tools) if tools else 0}, thinking={thinking_enabled}, rata_kanan={is_rata_kanan}")

    # Stats: request + model
    _STATS["requests"] += 1
    _STATS["model_usage"][model] = _STATS["model_usage"].get(model, 0) + 1

    # Cap thinking + answer output via max_completion_tokens if not set by caller
    # (prevents unbounded thinking bloat - research shows MiMo can produce 25-30k thinking tokens without cap)
    if not body.get("max_completion_tokens") and not body.get("max_tokens"):
        body["max_completion_tokens"] = 8192

    try:
        with open("last_request.json", "w", encoding="utf-8") as f_req:
            json.dump(body, f_req, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # Load authentic session cookies
    try:
        cookies = get_session_cookies()
    except Exception as e:
        _STATS["errors"] += 1
        return JSONResponse({"error": str(e)}, status_code=401)

    # Format unified prompt (with per-framework tool template)
    if tools or len(messages) > 1:
        agent_framework = detect_agent_framework(request, body)
        print(f"[mimoly] agent_framework={agent_framework} (profile: {AGENT_PROFILES[agent_framework]['label']})")
        user_prompt = build_agent_prompt(messages, tools, framework=agent_framework)
    else:
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                user_prompt = c if isinstance(c, str) else str(c)
                break

    if is_rata_kanan:
        amplifier = (
            "\n\n[REASONING ENGINE: MAXIMUM EFFORT / PENALARAN TERTINGGI (MENTOK)]\n"
            "Instruksi Berpikir & Analisis:\n"
            "- Maksimalkan kapasitas penalaran dan eksplorasi berpikir secara penuh (effort mentok).\n"
            "- Uraikan seluruh proses penalaran secara mendalam, teliti, terstruktur, dan kritis di dalam blok <think>.\n"
            "- Uji setiap premis, bandingkan alternatif pemikiran, verifikasi asumsi logika, dan evaluasi edge-cases sebelum menyimpulkan jawaban akhir atau memanggil tools.\n"
            "- Sajikan hasil akhir dengan akurasi, kedalaman, dan kualitas penalaran tertinggi."
        )
        user_prompt += amplifier

    ph_param = cookies.get("xiaomichatbot_ph", "")
    upstream_url = f"{CHAT_API_URL}?xiaomichatbot_ph={urllib.parse.quote(ph_param)}"

    headers = {
        "Content-Type": "application/json",
        "Accept-Language": "system",
        "x-timeZone": "Asia/Bangkok",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    # Ensure conversation is registered in Xiaomi database
    conv_id = uuid.uuid4().hex
    save_url = f"{CHAT_CONV_SAVE_URL}?xiaomichatbot_ph={urllib.parse.quote(ph_param)}"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=10.0) as save_client:
                await save_client.post(
                    save_url,
                    headers=headers,
                    cookies=cookies,
                    json={"conversationId": conv_id, "type": "chat", "title": f"Mimoly {target_upstream_model}"}
                )
            break
        except Exception as e:
            if attempt == 2:
                print(f"[mimoly] Warning: failed to save conversation: {e}")
            else:
                await asyncio.sleep(1.0)

    # Xiaomi payload structure
    upstream_payload = {
        "msgId": uuid.uuid4().hex,
        "conversationId": conv_id,
        "query": user_prompt,
        "isEditedQuery": False,
        "modelConfig": {
            "enableThinking": thinking_enabled,
            "webSearchStatus": "disabled",
            "model": target_upstream_model,
        },
        "multiMedias": []
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # --- STREAMING PATH ---
    if stream:
        async def sse_generator():
            # Initial role chunk only if tools are not requested
            if not tools:
                initial_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(initial_chunk)}\n\n"

            accumulated_chunks = []
            usage_data = None
            in_thinking = False

            try:
                for attempt in range(3):
                    try:
                        async with httpx.AsyncClient(timeout=180.0) as client:
                            async with client.stream("POST", upstream_url, headers=headers, cookies=cookies, json=upstream_payload) as resp:
                                if resp.status_code != 200:
                                    err_text = await resp.aread()
                                    err_chunk = {
                                        "id": completion_id,
                                        "object": "chat.completion.chunk",
                                        "created": created_time,
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": f"\n[Error from upstream: HTTP {resp.status_code}: {err_text.decode('utf-8', errors='ignore')}]"},
                                            "finish_reason": "error"
                                        }]
                                    }
                                    yield f"data: {json.dumps(err_chunk)}\n\n"
                                    yield "data: [DONE]\n\n"
                                    return

                                async for line in resp.aiter_lines():
                                    if not line:
                                        continue
                                    if line.startswith("data:"):
                                        print(f"[mimoly sse]: {line[:100]}")
                                    if not line.startswith("data:"):
                                        continue

                                    data_str = line[5:].strip()
                                    if "[DONE]" in data_str:
                                        print("[mimoly sse] Detected [DONE] marker")
                                        break

                                    try:
                                        parsed = json.loads(data_str)
                                    except Exception:
                                        continue
                                    # Ignore non-object frames (e.g. Xiaomi's internal
                                    # web-search JSON array) instead of crashing with
                                    # "'list' object has no attribute 'get'".
                                    parsed = normalize_upstream_frame(parsed)
                                    if parsed is None:
                                        continue

                                    if parsed.get("content") == "[DONE]":
                                        break

                                    # Capture usage metrics from upstream
                                    if "promptTokens" in parsed or "totalTokens" in parsed:
                                        usage_data = {
                                            "prompt_tokens": parsed.get("promptTokens", 0),
                                            "completion_tokens": parsed.get("completionTokens", 0),
                                            "total_tokens": parsed.get("totalTokens", 0),
                                        }
                                        native_usage = parsed.get("nativeUsage", {})
                                        reasoning_tokens = native_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                                        if reasoning_tokens:
                                            usage_data["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
                                        _record_usage(usage_data, model)
                                        continue

                                    content_piece = parsed.get("content", "")
                                    # Ignore dialog ID event
                                    if content_piece and content_piece.isdigit() and len(content_piece) >= 7:
                                        continue

                                    if not content_piece:
                                        continue

                                    if "<think>" in content_piece:
                                        in_thinking = True
                                        content_piece = content_piece.replace("<think>\x00", "").replace("<think>", "")

                                    if "</think>" in content_piece:
                                        parts = content_piece.split("</think>", 1)
                                        think_part = parts[0].replace("\x00", "")
                                        answer_part = parts[1].replace("\x00", "")

                                        if think_part and thinking_enabled:
                                            t_chunk = {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_time,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"role": "assistant", "reasoning_content": think_part},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(t_chunk)}\n\n"

                                        in_thinking = False

                                        if answer_part:
                                            if tools:
                                                accumulated_chunks.append(answer_part)
                                            else:
                                                chunk = {
                                                    "id": completion_id,
                                                    "object": "chat.completion.chunk",
                                                    "created": created_time,
                                                    "model": model,
                                                    "choices": [{
                                                        "index": 0,
                                                        "delta": {"role": "assistant", "content": answer_part},
                                                        "finish_reason": None
                                                    }]
                                                }
                                                yield f"data: {json.dumps(chunk)}\n\n"
                                                accumulated_chunks.append(answer_part)
                                        continue

                                    clean_piece = content_piece.replace("\x00", "")
                                    if not clean_piece:
                                        continue

                                    if in_thinking:
                                        if thinking_enabled:
                                            t_chunk = {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_time,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"role": "assistant", "reasoning_content": clean_piece},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(t_chunk)}\n\n"
                                    else:
                                        if tools:
                                            accumulated_chunks.append(clean_piece)
                                        else:
                                            chunk = {
                                                "id": completion_id,
                                                "object": "chat.completion.chunk",
                                                "created": created_time,
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"role": "assistant", "content": clean_piece},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {json.dumps(chunk)}\n\n"
                                            accumulated_chunks.append(clean_piece)
                        break
                    except Exception as conn_err:
                        if attempt == 2:
                            raise conn_err
                        print(f"[mimoly] Upstream stream retry {attempt+1}/3 due to: {conn_err}")
                        await asyncio.sleep(2.0)

            except Exception as e:
                err_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": f"\n[Stream connection error: {e}]"},
                        "finish_reason": "error"
                    }]
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return

            full_reply = "".join(accumulated_chunks)

            # If tools were active, parse tools and stream tool_calls
            if tools:
                raw_calls, clean_text = smart_extract_tool_calls(full_reply, user_prompt, tools)
                if raw_calls:
                    _STATS["tool_calls"] += len(raw_calls)
                    for tc in raw_calls:
                        tn = tc.get("name", "?")
                        _STATS["tool_usage"][tn] = _STATS["tool_usage"].get(tn, 0) + 1
                    openai_tool_calls = []
                    for i, tc in enumerate(raw_calls):
                        raw_args = tc.get("arguments", {})
                        args_str = json.dumps(raw_args) if isinstance(raw_args, dict) else str(raw_args)
                        openai_tool_calls.append({
                            "index": i,
                            "id": f"call_{int(time.time()*1000)}_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.get("name"),
                                "arguments": args_str
                            }
                        })
                    tc_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": openai_tool_calls
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(tc_chunk)}\n\n"

                    tc_finish = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "tool_calls"
                        }]
                    }
                    yield f"data: {json.dumps(tc_finish)}\n\n"
                else:
                    # Stream remaining clean text if no tool called
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": clean_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                    stop_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_time,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(stop_chunk)}\n\n"
            else:
                # Normal stop chunk
                stop_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(stop_chunk)}\n\n"

            # Final usage chunk
            if usage_data:
                usage_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "model": model,
                    "choices": [],
                    "usage": usage_data
                }
                yield f"data: {json.dumps(usage_chunk)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --- NON-STREAMING PATH ---
    accumulated_chunks = []
    usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    try:
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    async with client.stream("POST", upstream_url, headers=headers, cookies=cookies, json=upstream_payload) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            return JSONResponse(
                                {"error": f"Upstream HTTP {resp.status_code}: {err_body.decode('utf-8', errors='ignore')}"},
                                status_code=resp.status_code
                            )

                        async for line in resp.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if "[DONE]" in data_str:
                                break
                            try:
                                parsed = json.loads(data_str)
                            except Exception:
                                continue
                            # Ignore non-object frames (e.g. Xiaomi's internal web-search
                            # JSON array) instead of crashing with
                            # "'list' object has no attribute 'get'".
                            parsed = normalize_upstream_frame(parsed)
                            if parsed is None:
                                continue
                            if parsed.get("content") == "[DONE]":
                                break

                            if "promptTokens" in parsed or "totalTokens" in parsed:
                                usage_data = {
                                    "prompt_tokens": parsed.get("promptTokens", 0),
                                    "completion_tokens": parsed.get("completionTokens", 0),
                                    "total_tokens": parsed.get("totalTokens", 0),
                                }
                                native_usage = parsed.get("nativeUsage", {})
                                reasoning_tokens = native_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                                if reasoning_tokens:
                                    usage_data["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
                                _record_usage(usage_data, model)
                                continue

                            content_piece = parsed.get("content", "")
                            if content_piece and content_piece.isdigit() and len(content_piece) >= 7:
                                continue
                            if content_piece:
                                accumulated_chunks.append(content_piece.replace("\x00", ""))
                break
            except Exception as conn_err:
                if attempt == 2:
                    raise conn_err
                print(f"[mimoly] Upstream non-stream retry {attempt+1}/3 due to: {conn_err}")
                await asyncio.sleep(2.0)

    except Exception as e:
        _STATS["errors"] += 1
        return JSONResponse({"error": f"Failed to connect to upstream: {e}"}, status_code=502)

    full_reply = "".join(accumulated_chunks)

    # Separate thinking / reasoning tokens from content in non-streaming mode
    reasoning_text = None
    if "<think>" in full_reply:
        if "</think>" in full_reply:
            parts = full_reply.split("</think>", 1)
            reasoning_text = parts[0].replace("<think>", "").strip()
            full_reply = parts[1].strip()
        else:
            reasoning_text = full_reply.replace("<think>", "").strip()
            full_reply = ""

    # Detect tool calls
    raw_calls, clean_text = smart_extract_tool_calls(full_reply, user_prompt, tools)
    openai_tool_calls = []
    finish_reason = "stop"

    if raw_calls:
        finish_reason = "tool_calls"
        _STATS["tool_calls"] += len(raw_calls)
        for i, tc in enumerate(raw_calls):
            tn = tc.get("name", "?")
            _STATS["tool_usage"][tn] = _STATS["tool_usage"].get(tn, 0) + 1
            raw_args = tc.get("arguments", {})
            args_str = json.dumps(raw_args) if isinstance(raw_args, dict) else str(raw_args)
            openai_tool_calls.append({
                "id": f"call_{int(time.time()*1000)}_{i}",
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": args_str
                }
            })

    message_payload: Dict[str, Any] = {
        "role": "assistant",
        "content": clean_text if clean_text else (None if openai_tool_calls else full_reply),
    }
    if reasoning_text and thinking_enabled:
        message_payload["reasoning_content"] = reasoning_text

    if openai_tool_calls:
        message_payload["tool_calls"] = openai_tool_calls

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message_payload,
            "finish_reason": finish_reason
        }],
        "usage": usage_data
    }


# ==============================================================================
# CLI Commands
# ==============================================================================

def cmd_serve(args):
    """Run the pure Web2API proxy server."""
    print(f"[mimoly] Starting 100% Pure HTTP OpenAI-compatible proxy...")
    print(f"[mimoly] Zero Chrome processes, zero CDP overhead.")
    print(f"[mimoly] Endpoints:")
    print(f"  - Health:     http://{args.host}:{args.port}/health")
    print(f"  - Stats:      http://{args.host}:{args.port}/v1/stats")
    print(f"  - Dashboard:  http://{args.host}:{args.port}/dashboard")
    print(f"  - Models:     http://{args.host}:{args.port}/v1/models")
    print(f"  - Chat:       http://{args.host}:{args.port}/v1/chat/completions")
    
    # Verify session before starting
    try:
        cookies = get_session_cookies()
        print(f"[mimoly] Session loaded: {len(cookies)} cookies active (userId: {cookies.get('userId', 'unknown')})")
    except Exception as e:
        print(f"[mimoly] Warning: {e}")
        print(f"[mimoly] Run 'python mimoly.py login' if authentication fails.")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_test(args):
    """Perform a direct CLI test against Xiaomi MiMo via pure HTTP."""
    print(f"[mimoly] Testing pure HTTP connection to Xiaomi MiMo...")
    cookies = get_session_cookies()
    ph = cookies.get("xiaomichatbot_ph", "")
    url = f"{CHAT_API_URL}?xiaomichatbot_ph={urllib.parse.quote(ph)}"

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream, text/plain, */*",
        "Origin": "https://aistudio.xiaomimimo.com",
        "Referer": "https://aistudio.xiaomimimo.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    payload = {
        "msgId": uuid.uuid4().hex,
        "conversationId": uuid.uuid4().hex,
        "query": args.prompt,
        "modelConfig": {"enableThinking": True, "webSearchStatus": "disabled", "model": "mimo-v2.5-pro"},
        "multiMedias": []
    }

    async def _run():
        start_time = time.time()
        first_token = True
        ttft = 0.0
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, cookies=cookies, json=payload) as resp:
                print(f"[mimoly] HTTP Status: {resp.status_code}")
                if resp.status_code != 200:
                    body = await resp.aread()
                    print(f"[mimoly] Error response: {body.decode('utf-8', errors='ignore')}")
                    return

                print("[mimoly] Streaming response:")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            d = json.loads(data_str)
                            c = d.get("content", "")
                            if c and not (c.isdigit() and len(c) >= 7):
                                if first_token:
                                    ttft = time.time() - start_time
                                    first_token = False
                                print(c.replace("\x00", ""), end="", flush=True)
                        except Exception:
                            pass
        print(f"\n[mimoly] Completed in {time.time() - start_time:.2f}s (TTFT: {ttft:.3f}s)")

    asyncio.run(_run())


def cmd_login(args):
    """One-time interactive login: opens browser, extracts cookies without quotes, writes session.json."""
    import websockets

    def find_system_browser() -> str:
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\EdgeCore\152.0.4191.53\msedge.exe",
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "msedge",
        ]
        for c in candidates:
            if os.path.isabs(c):
                if os.path.isfile(c):
                    return c
            else:
                path = shutil.which(c)
                if path:
                    return path
        raise RuntimeError("No system Chrome or Edge browser found.")

    browser_bin = find_system_browser()
    temp_dir = tempfile.mkdtemp(prefix="mimoly_login_")
    cdp_port = 9223

    cmd = [
        browser_bin,
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={temp_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://aistudio.xiaomimimo.com/#/c"
    ]
    print(f"[mimoly] Launching login window ({browser_bin})...")
    print(f"[mimoly] Silakan login dengan akun Xiaomi Anda di jendela browser yang terbuka.")
    proc = subprocess.Popen(cmd)

    async def _capture():
        ws_url = None
        for _ in range(30):
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{cdp_port}/json") as r:
                    pages = json.loads(r.read().decode())
                    for p in pages:
                        if p.get("type") == "page":
                            ws_url = p.get("webSocketDebuggerUrl")
                            break
                    if ws_url:
                        break
            except Exception:
                pass
            await asyncio.sleep(1)

        if not ws_url:
            print("[mimoly] Could not connect to browser CDP.")
            return False

        async with websockets.connect(ws_url) as ws:
            print("[mimoly] Waiting for login credentials (max 300s)...")
            start = time.time()
            while time.time() - start < 300:
                await ws.send(json.dumps({"id": 1, "method": "Network.getCookies"}))
                msg = await ws.recv()
                data = json.loads(msg)
                cookies = data.get("result", {}).get("cookies", [])
                
                # Check target cookies
                mimo_cookies = [
                    c for c in cookies 
                    if c.get("name") in ["userId", "xiaomichatbot_ph", "xiaomichatbot_serviceToken"]
                ]
                has_ph = any(c["name"] == "xiaomichatbot_ph" for c in mimo_cookies)
                has_token = any(c["name"] == "xiaomichatbot_serviceToken" for c in mimo_cookies)

                if has_ph and has_token:
                    clean_cookies = []
                    for c in mimo_cookies:
                        clean_cookies.append({
                            "name": c["name"],
                            "value": str(c["value"]).strip('"'),
                            "domain": c.get("domain", ".xiaomimimo.com"),
                            "path": c.get("path", "/"),
                            "expires": c.get("expires", 0),
                        })
                    with open(SESSION_FILE, "w", encoding="utf-8") as f:
                        json.dump(clean_cookies, f, indent=2)
                    print(f"\n[mimoly] BERHASIL! Berhasil mengekstrak {len(clean_cookies)} cookies ke session.json.")
                    return True

                await asyncio.sleep(2)
                sys.stdout.write(".")
                sys.stdout.flush()

            print("\n[mimoly] Timeout waiting for login.")
            return False

    try:
        success = asyncio.run(_capture())
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            pass
        shutil.rmtree(temp_dir, ignore_errors=True)

    if success:
        print("[mimoly] Temporary browser directory cleaned up. You can now run: python mimoly.py serve")


def main():
    parser = argparse.ArgumentParser(description="Mimoly — 100% Pure HTTP OpenAI Proxy for Xiaomi MiMo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the proxy server")
    serve_parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind (default: 8080)")

    # test command
    test_parser = subparsers.add_parser("test", help="Test pure HTTP connection to MiMo")
    test_parser.add_argument("--prompt", default="Halo, siapa kamu?", help="Prompt to test")

    # login command
    login_parser = subparsers.add_parser("login", help="Interactive one-time browser login")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "login":
        cmd_login(args)


if __name__ == "__main__":
    main()
