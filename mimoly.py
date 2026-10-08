#!/usr/bin/env python3
"""
Mimoly — 100% Pure HTTP OpenAI-Compatible Web2API Proxy for Xiaomi MiMo Studio.
Zero browser downloads, zero headless Chrome running in background during proxy serve.
Ultra-lightweight, sub-second TTFT, native SSE streaming with exact token usage.
Compatible with OpenRouter, 9router, Hermes, OpenCode, Claude Code, and Cherry Studio.
"""

import argparse
import asyncio
import contextvars
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import FastAPI, Request, Response, UploadFile, File, Form
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

# Reliability primitives (error taxonomy, retry budget, account pool,
# admission control, exact TTL cache) — plain stdlib, no new dependencies.
# Ensure the module's own directory is importable when mimoly.py is loaded by
# path (e.g. importlib in the test suite or `hermes verify` from another CWD).
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if _MODULE_DIR not in sys.path:
    sys.path.insert(0, _MODULE_DIR)
try:
    from mimoly_reliability import (
        AdmissionController,
        AccountPool,
        PooledAccount,
        RateLimiter,
        RetryBudget,
        TTLCache,
        classify_exception,
        classify_status,
        is_retryable,
        should_rotate_key,
        ERROR_AUTH,
        ERROR_CLIENT,
        ERROR_GATEWAY,
        ERROR_NETWORK,
        ERROR_RATE_LIMIT,
        ERROR_SERVER,
    )
    RELIABILITY_AVAILABLE = True
except ImportError:
    RELIABILITY_AVAILABLE = False

    # Minimal no-op fallbacks so mimoly still runs if the helper module is absent.
    ERROR_AUTH = "auth"; ERROR_CLIENT = "client_error"; ERROR_GATEWAY = "gateway_error"
    ERROR_NETWORK = "network"; ERROR_RATE_LIMIT = "rate_limit"; ERROR_SERVER = "server_error"

    def classify_status(_code):  # type: ignore
        return None

    def classify_exception(_exc):  # type: ignore
        return ERROR_GATEWAY

    def is_retryable(_cls):  # type: ignore
        return False

    def should_rotate_key(_cls):  # type: ignore
        return False

    class RetryBudget:  # type: ignore
        def __init__(self, *a, **k): pass
        def start(self): pass
        def should_retry(self, attempt): return False
        def next_delay(self, attempt): return 0.0
        async def sleep(self, attempt): return None

    class AccountPool:  # type: ignore
        def __init__(self, *a, **k): self.accounts = []
        def acquire(self): return None
        def release(self, *a, **k): pass
        def healthy_count(self): return 0
        def snapshot(self): return []

    class AdmissionController:  # type: ignore
        def __init__(self, *a, **k): self.in_flight = 0; self.rejected = 0
        def acquire(self): return True
        def release(self): pass

    class RateLimiter:  # type: ignore
        def __init__(self, *a, **k): self.rejected = 0; self.allowed = 0
        def allow(self, key): return True
        def retry_after(self, key): return 0.0
        def stats(self): return {}

    class TTLCache:  # type: ignore
        def __init__(self, *a, **k): pass
        def make_key(self, body): return ""
        def get(self, key): return None
        def set(self, key, value): pass
        def stats(self): return {}


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


UPSTREAM_BUSY_PHRASES = (
    "服务器繁忙",
    "系统繁忙",
    "服务繁忙",
    "请稍后再试",
    "请稍后重试",
    "请求过于频繁",
    "Please do not submit repeatedly",
    "do not submit repeatedly",
    "The server is busy",
    "Server is busy",
    "Too many requests",
    "Please try again later",
    "Service is temporarily unavailable",
    "query is too long",
    "query too long",
    "Hello, I cannot answer this question at the moment",
    "let's talk about something else instead",
    "我现在无法回答这个问题",
)


def is_upstream_busy(text: str) -> bool:
    """Return True if text contains any upstream capacity/overload error phrases."""
    if not text:
        return False
    return any(p in text for p in UPSTREAM_BUSY_PHRASES)


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
SESSION_FILE = Path(os.environ.get("MIMOLY_SESSION_FILE", BASE_DIR / "session.json")).resolve()
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
CHAT_API_URL = "https://aistudio.xiaomimimo.com/open-apis/bot/chat"
CHAT_CONV_SAVE_URL = "https://aistudio.xiaomimimo.com/open-apis/chat/conversation/save"

# ---------------------------------------------------------------------------
# Reliability & scale knobs (env-overridable)
# ---------------------------------------------------------------------------
# One shared AsyncClient => connection pooling + TLS session reuse across ALL
# requests (previously a fresh client per request wasted handshakes).
_MIMOLY_MAX_CONNECTIONS = int(os.environ.get("MIMOLY_MAX_CONNECTIONS", "200"))
_MIMOLY_MAX_KEEPALIVE = int(os.environ.get("MIMOLY_MAX_KEEPALIVE", "50"))
_MIMOLY_ADMISSION_LIMIT = int(os.environ.get("MIMOLY_MAX_INFLIGHT", "0"))  # 0 = disabled
_MIMOLY_CACHE_TTL = float(os.environ.get("MIMOLY_CACHE_TTL", "300"))
_MIMOLY_CACHE_MAX = int(os.environ.get("MIMOLY_CACHE_MAXSIZE", "256"))
_MIMOLY_ACCOUNTS_FILE = os.environ.get("MIMOLY_ACCOUNTS_FILE", "")
_MIMOLY_RATE_LIMIT = int(os.environ.get("MIMOLY_RATE_LIMIT", "0"))          # per-IP req/window; 0 = off
_MIMOLY_RATE_WINDOW = float(os.environ.get("MIMOLY_RATE_WINDOW", "60"))     # seconds

_HTTP_LIMITS = httpx.Limits(
    max_connections=_MIMOLY_MAX_CONNECTIONS,
    max_keepalive_connections=_MIMOLY_MAX_KEEPALIVE,
)
# Upstream MiMo streams can run >60s during deep thinking; a read timeout is
# applied per-chunk via asyncio.wait_for in the stream loop, so the client-level
# timeout stays generous while still bounding a truly dead connection.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)

_SHARED_CLIENT: Optional["httpx.AsyncClient"] = None


def get_shared_client() -> "httpx.AsyncClient":
    """Return the process-wide AsyncClient (lazily created, connection-pooled)."""
    global _SHARED_CLIENT
    if _SHARED_CLIENT is None or _SHARED_CLIENT.is_closed:
        _SHARED_CLIENT = httpx.AsyncClient(
            timeout=_UPSTREAM_TIMEOUT,
            limits=_HTTP_LIMITS,
            follow_redirects=False,
        )
    return _SHARED_CLIENT


async def close_shared_client() -> None:
    global _SHARED_CLIENT
    if _SHARED_CLIENT is not None and not _SHARED_CLIENT.is_closed:
        try:
            await _SHARED_CLIENT.aclose()
        except Exception:
            pass
    _SHARED_CLIENT = None


# Account pool (multi-credential rotation). Populated at startup from
# MIMOLY_ACCOUNTS_FILE (JSON list of {id, cookies}) and/or session.json.
_ACCOUNT_POOL: Optional["AccountPool"] = None
_ADMISSION = AdmissionController(_MIMOLY_ADMISSION_LIMIT)
_RESPONSE_CACHE = TTLCache(maxsize=_MIMOLY_CACHE_MAX, ttl=_MIMOLY_CACHE_TTL)
_RATE_LIMITER = RateLimiter(limit=_MIMOLY_RATE_LIMIT, window=_MIMOLY_RATE_WINDOW)

# Per-request retry counter, surfaced to callers via the X-Retry-Attempt-Count
# response header (non-streaming) so a rising retry rate is visible.
_RETRY_ATTEMPTS: "contextvars.ContextVar[int]" = contextvars.ContextVar("mimoly_retry_attempts", default=0)


def _bump_retry() -> None:
    """Record one retry for the current request (also feeds the global counter)."""
    try:
        _RETRY_ATTEMPTS.set(_RETRY_ATTEMPTS.get() + 1)
    except Exception:
        pass
    _STATS["retries"] = _STATS.get("retries", 0) + 1


def load_account_pool() -> Optional["AccountPool"]:
    """Build the multi-account pool from a JSON file or the single session.json.

    File format (``MIMOLY_ACCOUNTS_FILE``): a JSON list of accounts, each either
    ``{"id": "...", "cookies": {...}}`` or a flat cookie dict. When the file is
    absent we fall back to the single session.json so behaviour is unchanged.
    """
    global _ACCOUNT_POOL
    accounts: List[Dict[str, Any]] = []

    if _MIMOLY_ACCOUNTS_FILE and Path(_MIMOLY_ACCOUNTS_FILE).exists():
        try:
            with open(_MIMOLY_ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "accounts" in raw:
                raw = raw["accounts"]
            if isinstance(raw, list):
                for i, item in enumerate(raw):
                    if not isinstance(item, dict):
                        continue
                    acct_id = str(item.get("id") or item.get("userId") or f"acct{i+1}")
                    cookies = item.get("cookies") if isinstance(item.get("cookies"), dict) else item
                    cookies = {k: str(v).strip('"') for k, v in cookies.items() if k != "id"}
                    accounts.append({"id": acct_id, "cookies": cookies})
        except Exception as e:
            print(f"[mimoly] Warning: failed to load accounts file {_MIMOLY_ACCOUNTS_FILE}: {e}")

    if not accounts:
        try:
            cookies = get_session_cookies()
            accounts = [{"id": cookies.get("userId", "default"), "cookies": cookies}]
        except Exception:
            accounts = []

    _ACCOUNT_POOL = AccountPool(
        accounts,
        max_failures=int(os.environ.get("MIMOLY_ACCOUNT_MAX_FAILURES", "3")),
        cooldown_base=float(os.environ.get("MIMOLY_ACCOUNT_COOLDOWN_BASE", "30")),
        cooldown_max=float(os.environ.get("MIMOLY_ACCOUNT_COOLDOWN_MAX", "1800")),
    )
    return _ACCOUNT_POOL


def get_account_pool() -> Optional["AccountPool"]:
    return _ACCOUNT_POOL


_FALLBACK_ACCOUNT: Optional[PooledAccount] = None


def acquire_account() -> Tuple[Dict[str, str], Optional[Any]]:
    """Return (cookies, pooled_account). Falls back to session.json cookies when
    the pool is empty or every account is cooling down."""
    global _FALLBACK_ACCOUNT
    pool = _ACCOUNT_POOL
    if pool and pool.accounts:
        acct = pool.acquire()
        if acct is not None:
            return acct.cookies, acct
        # Pool exhausted (all cooling down): last-resort single-session cookies.
        print("[mimoly] Warning: all pooled accounts are cooling down; using session.json fallback.")
    cookies = get_session_cookies()
    if _FALLBACK_ACCOUNT is None or _FALLBACK_ACCOUNT.cookies != cookies:
        _FALLBACK_ACCOUNT = PooledAccount("fallback", cookies)
    return cookies, _FALLBACK_ACCOUNT


def release_account(acct: Optional[Any], success: bool, error_class: Optional[str] = None) -> None:
    if _ACCOUNT_POOL is not None and acct is not None:
        _ACCOUNT_POOL.release(acct, success=success, error_class=error_class)
        if hasattr(acct, "get_lock"):
            try:
                l = acct.get_lock()
                if l.locked():
                    l.release()
            except RuntimeError:
                pass


def can_rotate_account() -> bool:
    """True if the pool holds more than one credential, so retrying on another
    account is meaningful (e.g. after a key-specific 401/403)."""
    pool = _ACCOUNT_POOL
    if pool is None or len(pool.accounts) <= 1:
        return False
    return pool.healthy_count() >= 1


def get_upstream_endpoints(model: str, ph: str) -> Tuple[str, str]:
    """Resolve upstream chat and conversation save URLs based on model cluster."""
    is_ultraspeed = (model in ("mimo-v2.6-pro-ultraspeed-studio", "mimo-v2.5-pro-ultraspeed", "mimo-v2.6-pro-ultraspeed", "mimo-ultraspeed"))
    prefix = "https://aistudio.xiaomimimo.com/fastchat" if is_ultraspeed else "https://aistudio.xiaomimimo.com"
    quoted_ph = urllib.parse.quote(ph)
    chat_url = f"{prefix}/open-apis/bot/chat?xiaomichatbot_ph={quoted_ph}"
    save_url = f"{prefix}/open-apis/chat/conversation/save?xiaomichatbot_ph={quoted_ph}"
    return chat_url, save_url

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
    {
        "id": "mimo-v2.6-pro-ultraspeed-studio",
        "name": "MiMo-V2.6-Pro-UltraSpeed (Fast Multimodal Reasoning)",
        "upstream_model": "mimo-v2.5-pro-ultraspeed",
        "description": "Xiaomi fast-response multimodal model with deep thinking capabilities."
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
    # UltraSpeed
    "mimo-v2.6-pro-ultraspeed": "mimo-v2.5-pro-ultraspeed",
    "mimo-v2.6-pro-ultraspeed-studio": "mimo-v2.5-pro-ultraspeed",
    "mimo-ultraspeed": "mimo-v2.5-pro-ultraspeed",
    "mimo-v2.5-pro-ultraspeed": "mimo-v2.5-pro-ultraspeed",
    # Flash / Fast
    "mimo-v2.6-flash": "mimo-v2.6-flash",
    "mimo-v2.5": "mimo-v2.6-flash",
    "mimo-v2-flash": "mimo-v2.6-flash",
    "mimo-v2.1-omni": "mimo-v2.6-flash",
    "mimo-flash": "mimo-v2.6-flash",
}

VOICE_MAP = {
    # OpenAI standard voice aliases mapped to Xiaomi MiMo TTS voices
    "alloy": "Mia",
    "echo": "Dean",
    "fable": "Chloe",
    "onyx": "Milo",
    "nova": "bingtang",
    "shimmer": "moli",
    # English names
    "mia": "Mia",
    "chloe": "Chloe",
    "milo": "Milo",
    "dean": "Dean",
    "bingtang": "冰糖",
    "moli": "茉莉",
    "suda": "苏打",
    "baihua": "白桦",
    # Chinese names
    "冰糖": "冰糖",
    "茉莉": "茉莉",
    "苏打": "苏打",
    "白桦": "白桦",
    "Mia": "Mia",
    "Chloe": "Chloe",
    "Milo": "Milo",
    "Dean": "Dean",
}


def sync_upstream_models(config_data: dict) -> int:
    """Dynamically register models from Xiaomi bot/config into catalog and aliases."""
    added = 0
    model_list = config_data.get("modelConfigList") or config_data.get("data", {}).get("modelConfigList", [])
    if not isinstance(model_list, list):
        return 0

    existing_ids = {m["id"] for m in MODEL_CATALOG}
    for item in model_list:
        if not isinstance(item, dict):
            continue
        model_id = item.get("model")
        if not model_id:
            continue
        name = item.get("name") or model_id
        desc = item.get("enIntro") or item.get("cnIntro") or f"Xiaomi {name} model"
        
        if model_id not in existing_ids:
            MODEL_CATALOG.append({
                "id": model_id,
                "name": name,
                "upstream_model": model_id,
                "description": desc
            })
            existing_ids.add(model_id)
            added += 1

        if model_id not in MODEL_ALIASES:
            MODEL_ALIASES[model_id] = model_id
            
        # Clean suffix alias
        clean_alias = model_id.replace("-studio", "")
        if clean_alias not in MODEL_ALIASES:
            MODEL_ALIASES[clean_alias] = model_id

    return added

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
    "finish_reasons": {},    # {stop|tool_calls|length|error: count}
    "latency_ms": {"count": 0, "sum": 0.0, "min": None, "max": None},
    "ttft_ms": {"count": 0, "sum": 0.0, "min": None, "max": None},  # time to first token (stream)
    "timeline": {},          # {minute_epoch: {requests, errors, tool_calls, total_tokens}}
    "cache_hits": 0,         # exact-match cache hits (non-streaming)
}


def _record_latency(bucket: str, ms: float) -> None:
    """Accumulate a latency sample into min/max/sum/count."""
    b = _STATS[bucket]
    b["count"] += 1
    b["sum"] += ms
    b["min"] = ms if b["min"] is None else min(b["min"], ms)
    b["max"] = ms if b["max"] is None else max(b["max"], ms)


def _record_finish_reason(reason: str) -> None:
    """Count how each request finished (stop / tool_calls / length / error)."""
    if not reason:
        return
    _STATS["finish_reasons"][reason] = _STATS["finish_reasons"].get(reason, 0) + 1


def _record_timeline(requests: int = 0, errors: int = 0,
                     tool_calls: int = 0, total_tokens: int = 0) -> None:
    """Bucket activity by wall-clock minute (keeps the last ~2h for the line chart)."""
    minute = int(time.time() // 60) * 60
    tl = _STATS["timeline"]
    bucket = tl.setdefault(
        minute, {"requests": 0, "errors": 0, "tool_calls": 0, "total_tokens": 0}
    )
    bucket["requests"] += requests
    bucket["errors"] += errors
    bucket["tool_calls"] += tool_calls
    bucket["total_tokens"] += total_tokens
    if len(tl) > 120:  # prune oldest beyond 120 minutes
        for k in sorted(tl.keys())[:-120]:
            del tl[k]


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
    _record_timeline(total_tokens=tt)
    if model:
        mt = _STATS["model_tokens"].setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        mt["prompt_tokens"] += pt
        mt["completion_tokens"] += ct
        mt["total_tokens"] += tt


# ---------------------------------------------------------------------------
# Per-request log (bounded ring buffer of the last N requests, newest last).
# Shows WHAT was asked (input prompt) and WHAT came back (output), so the
# dashboard is useful for debugging, not just aggregate counters.
# ---------------------------------------------------------------------------
_REQUEST_LOG_MAX = 50
_REQUEST_LOG: "deque[Dict[str, Any]]" = deque(maxlen=_REQUEST_LOG_MAX)


def _clip(text: Any, limit: int = 400) -> str:
    """Coerce to str and truncate for display; marks truncation with an ellipsis."""
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[:limit].rstrip() + "…"


def _last_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user message text (handles multimodal content lists)."""
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
            return " ".join(p for p in parts if p)
        return str(c)
    return ""


def _new_request(model: str, stream: bool, messages: int, tools: int,
                 prompt: str, tools_names: List[str]) -> Dict[str, Any]:
    """Create + register a request-log record; returns it so it can be finalized."""
    rec = {
        "id": uuid.uuid4().hex[:10],
        "ts": time.time(),
        "model": model,
        "stream": stream,
        "messages": messages,
        "tools": tools,
        "tools_names": list(tools_names or []),
        "input": _clip(prompt, 400),
        "output": "",
        "finish_reason": "",
        "latency_ms": None,
        "ttft_ms": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tools_called": [],
    }
    _REQUEST_LOG.append(rec)
    return rec


def _finish_request(rec: Optional[Dict[str, Any]], output: str = "", finish_reason: str = "",
                    latency_ms: Optional[float] = None, ttft_ms: Optional[float] = None,
                    usage: Optional[Dict[str, Any]] = None,
                    tools_called: Optional[List[str]] = None) -> None:
    """Finalize a request-log record with its outcome (idempotent, never raises)."""
    if rec is None:
        return
    try:
        if output:
            rec["output"] = _clip(output, 600)
        if finish_reason:
            rec["finish_reason"] = finish_reason
        if latency_ms is not None:
            rec["latency_ms"] = round(latency_ms, 1)
        if ttft_ms is not None:
            rec["ttft_ms"] = round(ttft_ms, 1)
        if usage:
            rec["prompt_tokens"] = usage.get("prompt_tokens", 0) or 0
            rec["completion_tokens"] = usage.get("completion_tokens", 0) or 0
            rec["total_tokens"] = usage.get("total_tokens", 0) or 0
        if tools_called is not None:
            rec["tools_called"] = list(tools_called)
    except Exception:
        pass


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


def extract_images_from_messages(messages: list) -> List[Dict[str, Any]]:
    """Extract image data (base64 or URL) from OpenAI-format messages."""
    images = []
    seen = set()
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "image_url":
                    img_info = part.get("image_url", {})
                    url = img_info.get("url", "") if isinstance(img_info, dict) else str(img_info)
                    if not url:
                        continue
                    if url.startswith("data:"):
                        try:
                            header, b64_str = url.split(",", 1)
                            mime = header.split(";")[0].split(":")[1] if ":" in header else "image/jpeg"
                            if b64_str and b64_str not in seen:
                                seen.add(b64_str)
                                images.append({
                                    "type": "base64",
                                    "base64": b64_str,
                                    "mime_type": mime,
                                    "url": url
                                })
                        except Exception:
                            pass
                    elif url.startswith("http://") or url.startswith("https://"):
                        if url not in seen:
                            seen.add(url)
                            images.append({
                                "type": "url",
                                "url": url,
                                "mime_type": "image/jpeg"
                            })
    return images


async def upload_media_to_mimo(
    base64_data: str,
    mime_type: str,
    cookies: dict,
    ph: str,
    model: str = "mimo-v2.6-pro"
) -> Optional[Dict[str, Any]]:
    """Upload image/media to Xiaomi MiMo OSS storage via 3-step flow."""
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    import base64 as b64
    try:
        binary_data = b64.b64decode(base64_data)
    except Exception as e:
        print(f"[mimoly] base64 decode failed: {e}")
        return None

    ext = mime_type.split("/")[-1] if "/" in mime_type else "jpg"
    if ext == "jpeg":
        ext = "jpg"
    file_name = f"{uuid.uuid4().hex[:16]}.{ext}"

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://aistudio.xiaomimimo.com/",
        "Origin": "https://aistudio.xiaomimimo.com"
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            info_res = await client.post(
                "https://aistudio.xiaomimimo.com/open-apis/resource/genUploadInfo",
                params={"xiaomichatbot_ph": ph},
                json={"fileName": file_name},
                headers=headers,
                cookies=cookies
            )
            info_data = info_res.json()
            if info_data.get("code") != 0 or not info_data.get("data"):
                print(f"[mimoly] genUploadInfo failed: {info_data}")
                return None

            upload_url = info_data["data"]["uploadUrl"]
            resource_url = info_data["data"]["resourceUrl"]
            object_name = info_data["data"]["objectName"]

            put_headers = {"Content-Type": "application/octet-stream"}
            put_res = await client.put(upload_url, content=binary_data, headers=put_headers)
            if put_res.status_code != 200:
                print(f"[mimoly] PUT to OSS failed: {put_res.status_code}")
                return None

            parse_params = {
                "fileUrl": resource_url,
                "objectName": object_name,
                "model": model,
                "xiaomichatbot_ph": ph,
            }

            parse_res = None
            for attempt in range(5):
                try:
                    resp = await client.post(
                        "https://aistudio.xiaomimimo.com/open-apis/resource/parse",
                        params=parse_params,
                        json={},
                        headers=headers,
                        cookies=cookies
                    )
                    data = resp.json()
                    if data.get("code") == 0 and data.get("data", {}).get("id"):
                        parse_res = data
                        break
                except Exception:
                    pass
                await asyncio.sleep(1.0)

            if not parse_res:
                print("[mimoly] Resource parse failed after retries")
                return None

            resource_id = parse_res["data"]["id"]
            return {
                "mediaType": "image",
                "fileUrl": resource_url,
                "compressedVideoUrl": "",
                "audioTrackUrl": "",
                "name": file_name,
                "size": len(binary_data),
                "status": "completed",
                "objectName": object_name,
                "tokenUsage": parse_res["data"].get("tokenUsage", 106),
                "url": resource_id
            }
    except Exception as e:
        print(f"[mimoly] upload_media_to_mimo exception: {e}")
        return None


async def prepare_multimedias_for_request(
    messages: list,
    cookies: dict,
    ph: str,
    model: str = "mimo-v2.6-pro"
) -> List[Dict[str, Any]]:
    """Detect images in messages and upload to Xiaomi OSS for multiMedias field."""
    images = extract_images_from_messages(messages)
    if not images or not ph:
        return []

    multi_medias = []
    for img in images:
        b64_str = img.get("base64")
        mime = img.get("mime_type", "image/jpeg")
        if not b64_str and img.get("type") == "url":
            try:
                async with httpx.AsyncClient(timeout=15.0) as dl_client:
                    r = await dl_client.get(img["url"])
                    if r.status_code == 200:
                        import base64 as b64
                        b64_str = b64.b64encode(r.content).decode("utf-8")
                        mime = r.headers.get("content-type", mime)
            except Exception as e:
                print(f"[mimoly] Failed to fetch remote image {img['url']}: {e}")
                continue

        if b64_str:
            media_obj = await upload_media_to_mimo(b64_str, mime, cookies, ph, model=model)
            if media_obj:
                multi_medias.append(media_obj)

    return multi_medias


def convert_anthropic_messages(messages: list, system: Optional[str] = None) -> list:
    """Convert Anthropic Messages API messages into OpenAI Chat Completions format."""
    result = []
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, list):
            text_parts = []
            tool_calls = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                b_type = block.get("type")
                if b_type == "text":
                    text_parts.append(block.get("text", ""))
                elif b_type == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"tu_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False) if isinstance(block.get("input"), dict) else str(block.get("input", ""))
                        }
                    })
                elif b_type == "thinking":
                    text_parts.append(block.get("thinking", ""))
            combined = "\n".join(t for t in text_parts if t)
            obj = {"role": "assistant", "content": combined or None}
            if tool_calls:
                obj["tool_calls"] = tool_calls
            result.append(obj)

        elif role == "user" and isinstance(content, list):
            new_blocks = []
            text_parts = []
            tool_results = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                b_type = block.get("type")
                if b_type == "text":
                    text_parts.append(block.get("text", ""))
                elif b_type == "image":
                    src = block.get("source", {})
                    img_type = src.get("media_type", "image/png")
                    img_data = src.get("data", "")
                    if src.get("type") == "base64":
                        new_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{img_type};base64,{img_data}"}
                        })
                    elif src.get("type") == "url":
                        new_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": src.get("url", "")}
                        })
                elif b_type == "tool_result":
                    tool_results.append(block)

            if tool_results:
                for tr in tool_results:
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, list):
                        tr_text = " ".join(
                            b.get("text", "") for b in tr_content
                            if isinstance(b, dict) and b.get("type") == "text"
                        )
                    else:
                        tr_text = str(tr_content) if tr_content else ""
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr_text,
                    })

            if not tool_results:
                combined_text = "\n".join(t for t in text_parts if t)
                if combined_text:
                    new_blocks.insert(0, {"type": "text", "text": combined_text})
                if new_blocks:
                    result.append({"role": "user", "content": new_blocks})
                else:
                    result.append({"role": "user", "content": combined_text})
        elif isinstance(content, str):
            result.append({"role": role, "content": content})
        else:
            result.append({"role": role, "content": str(content) if content else ""})

    return result


def convert_anthropic_tools(tools: Optional[list]) -> Optional[list]:
    """Convert Anthropic tools list to OpenAI function calling format."""
    if not tools:
        return None
    result = []
    for t in tools:
        if isinstance(t, dict):
            fn = {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
            result.append({"type": "function", "function": fn})
    return result if result else None


def convert_anthropic_request(body: dict) -> dict:
    """Convert Anthropic /v1/messages request to OpenAI /v1/chat/completions format."""
    model = body.get("model", "mimo-v2.6-pro")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens")
    system = body.get("system", None)
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    temperature = body.get("temperature", None)
    top_p = body.get("top_p", None)
    stop_sequences = body.get("stop_sequences", None)

    openai_msgs = convert_anthropic_messages(messages, system)
    openai_tools = convert_anthropic_tools(tools)

    res = {
        "model": model,
        "messages": openai_msgs,
        "stream": stream,
    }
    if max_tokens is not None:
        res["max_tokens"] = max_tokens
    if openai_tools:
        res["tools"] = openai_tools
    if temperature is not None:
        res["temperature"] = temperature
    if top_p is not None:
        res["top_p"] = top_p
    if stop_sequences:
        res["stop"] = stop_sequences
    return res


def convert_openai_to_anthropic_response(openai_body: dict, model: str, msg_id: Optional[str] = None) -> dict:
    """Convert OpenAI chat completion response to Anthropic /v1/messages response format."""
    msg_id = msg_id or f"msg_{uuid.uuid4().hex[:24]}"
    choices = openai_body.get("choices", [{}])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    content = message.get("content", "") or ""
    reasoning = message.get("reasoning_content", "") or ""
    tool_calls = message.get("tool_calls", None)

    content_blocks = []
    if reasoning:
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": ""
        })

    text = content.strip()
    if tool_calls:
        if text:
            content_blocks.append({"type": "text", "text": text})
        stop_reason = "tool_use"
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"tu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": args
            })
    else:
        stop_reason = "end_turn"
        if text or not reasoning:
            content_blocks.append({"type": "text", "text": text})

    usage_data = openai_body.get("usage", {})
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage_data.get("prompt_tokens", 0),
            "output_tokens": usage_data.get("completion_tokens", 0)
        }
    }


def convert_responses_request_to_chat(body: dict) -> dict:
    """Convert OpenAI Responses API request format to Chat Completions request format."""
    model = body.get("model", "mimo-v2.6-pro")
    messages = []

    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    inp = body.get("input", "")
    if isinstance(inp, str):
        if inp:
            messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        for item in inp:
            if isinstance(item, dict):
                r = item.get("role") or item.get("type", "user")
                c = item.get("content", "")
                if r in ("system", "user", "assistant", "tool"):
                    messages.append({"role": r, "content": c})
                elif item.get("type") == "message":
                    messages.append({"role": item.get("role", "user"), "content": c})
                elif item.get("type") == "function_call_output":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": str(item.get("output", ""))
                    })
            elif isinstance(item, str):
                messages.append({"role": "user", "content": item})

    chat_body = {
        "model": model,
        "messages": messages,
        "stream": body.get("stream", False),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
    }
    tools = body.get("tools")
    if tools:
        chat_tools = []
        for t in tools:
            if isinstance(t, dict):
                if t.get("type") == "function" and "function" in t:
                    chat_tools.append(t)
                elif "name" in t:
                    chat_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name"),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters") or t.get("input_schema", {})
                        }
                    })
        if chat_tools:
            chat_body["tools"] = chat_tools

    return chat_body


def convert_chat_to_responses_output(chat_resp: dict, model: str) -> dict:
    """Convert OpenAI Chat Completions response to OpenAI Responses API output format."""
    resp_id = f"resp_{uuid.uuid4().hex[:20]}"
    created_at = chat_resp.get("created", int(time.time()))
    choices = chat_resp.get("choices", [{}])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})

    content = message.get("content", "") or ""
    tool_calls = message.get("tool_calls", [])

    output_items = []
    if content:
        output_items.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:20]}",
            "role": "assistant",
            "content": [{"type": "text", "text": content}]
        })

    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function", {})
            output_items.append({
                "type": "function_call",
                "id": tc.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:16]}"),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "{}")
            })

    return {
        "id": resp_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": model,
        "output": output_items,
        "usage": chat_resp.get("usage", {})
    }


_COOKIE_HEALTH: Dict[str, Any] = {
    "valid": True,
    "status": "unverified",
    "last_checked": 0,
    "message": ""
}


async def check_cookie_health(cookies: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Probe Xiaomi upstream to verify if session cookies are active or expired."""
    global _COOKIE_HEALTH
    if cookies is None:
        try:
            cookies = get_session_cookies()
        except Exception as e:
            _COOKIE_HEALTH = {
                "valid": False,
                "status": "missing",
                "last_checked": int(time.time()),
                "message": str(e)
            }
            return _COOKIE_HEALTH

    ph = cookies.get("xiaomichatbot_ph", "")
    if not ph:
        _COOKIE_HEALTH = {
            "valid": False,
            "status": "missing_ph",
            "last_checked": int(time.time()),
            "message": "Missing xiaomichatbot_ph in session"
        }
        return _COOKIE_HEALTH

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://aistudio.xiaomimimo.com/",
        "Origin": "https://aistudio.xiaomimimo.com"
    }
    url = f"https://aistudio.xiaomimimo.com/open-apis/bot/config?xiaomichatbot_ph={urllib.parse.quote(ph)}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, cookies=cookies)
            if resp.status_code == 200:
                _COOKIE_HEALTH = {
                    "valid": True,
                    "status": "active",
                    "last_checked": int(time.time()),
                    "message": "Session cookies valid and active"
                }
            elif resp.status_code in (401, 403):
                _COOKIE_HEALTH = {
                    "valid": False,
                    "status": "expired",
                    "last_checked": int(time.time()),
                    "message": f"Session expired (HTTP {resp.status_code}). Please refresh session.json via 'python mimoly.py login'"
                }
                print(f"[mimoly] [HARVESTER ALERT] {_COOKIE_HEALTH['message']}")
            else:
                _COOKIE_HEALTH = {
                    "valid": False,
                    "status": f"http_{resp.status_code}",
                    "last_checked": int(time.time()),
                    "message": f"Upstream returned HTTP {resp.status_code}"
                }
    except Exception as e:
        _COOKIE_HEALTH = {
            "valid": False,
            "status": "connection_error",
            "last_checked": int(time.time()),
            "message": str(e)
        }

    return _COOKIE_HEALTH


async def tts_generate_audio(
    text: str,
    voice: str = "alloy",
    model: str = "mimo-v2.5-tts",
    cookies: Optional[dict] = None
) -> bytes:
    """Generate speech audio via Xiaomi TTS backend."""
    if cookies is None:
        cookies = get_session_cookies()
    ph = cookies.get("xiaomichatbot_ph", "")
    mimo_voice = VOICE_MAP.get(voice, "default_zh")

    conversation_id = uuid.uuid4().hex[:32]
    msg_id = uuid.uuid4().hex[:32]
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://aistudio.xiaomimimo.com/",
        "Origin": "https://aistudio.xiaomimimo.com"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        save_url = f"https://aistudio.xiaomimimo.com/open-apis/chat/conversation/save?xiaomichatbot_ph={urllib.parse.quote(ph)}"
        await client.post(
            save_url,
            headers=headers,
            cookies=cookies,
            json={"conversationId": conversation_id, "title": "TTS", "type": "tts"}
        )

        gen_url = f"https://aistudio.xiaomimimo.com/open-apis/tts/v2/generate?xiaomichatbot_ph={urllib.parse.quote(ph)}"
        payload = {
            "conversationId": conversation_id,
            "msgId": msg_id,
            "content": {
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "assistant", "content": text}
                ],
                "audio": {"format": "wav", "voice": mimo_voice}
            },
            "modelConfig": {"modelCode": "mimo-v2.5-tts", "scene": "BRIEF_DESCRIPTION"}
        }
        res = await client.post(gen_url, headers=headers, cookies=cookies, json=payload)
        if res.status_code != 200:
            raise RuntimeError(f"TTS request failed HTTP {res.status_code}: {res.text}")
        data = res.json()
        if data.get("code") != 0 or not data.get("data"):
            raise RuntimeError(f"TTS error: {data.get('msg', 'unknown error')}")
        task_id = data["data"].get("taskId")

        status_url = f"https://aistudio.xiaomimimo.com/open-apis/tts/generateStatus?xiaomichatbot_ph={urllib.parse.quote(ph)}&taskId={urllib.parse.quote(str(task_id))}"
        audio_url = None
        for _ in range(45):
            await asyncio.sleep(1.0)
            st_res = await client.get(status_url, headers=headers, cookies=cookies)
            if st_res.status_code == 200:
                st_data = st_res.json()
                if st_data.get("code") == 0 and st_data.get("data"):
                    st = st_data["data"].get("status")
                    if st == "success":
                        audio_url = st_data["data"].get("audioUrl")
                        break
                    elif st == "failed":
                        raise RuntimeError("TTS generation failed upstream")

        if not audio_url:
            raise TimeoutError("TTS generation timed out waiting for audio URL")

        dl_res = await client.get(audio_url)
        return dl_res.content


async def asr_transcribe_audio(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: str = "auto",
    cookies: Optional[dict] = None
) -> str:
    """Transcribe audio via Xiaomi ASR backend."""
    if cookies is None:
        cookies = get_session_cookies()
    ph = cookies.get("xiaomichatbot_ph", "")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://aistudio.xiaomimimo.com/",
        "Origin": "https://aistudio.xiaomimimo.com"
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        info_res = await client.post(
            f"https://aistudio.xiaomimimo.com/open-apis/resource/genUploadInfo?xiaomichatbot_ph={urllib.parse.quote(ph)}",
            headers=headers,
            cookies=cookies,
            json={"fileName": filename}
        )
        info_data = info_res.json()
        if info_data.get("code") != 0 or not info_data.get("data"):
            raise RuntimeError(f"ASR genUploadInfo failed: {info_data}")

        upload_url = info_data["data"]["uploadUrl"]
        resource_url = info_data["data"]["resourceUrl"]

        put_res = await client.put(upload_url, content=audio_bytes, headers={"Content-Type": "application/octet-stream"})
        if put_res.status_code != 200:
            raise RuntimeError(f"ASR PUT failed: {put_res.status_code}")

        conv_id = uuid.uuid4().hex[:32]
        msg_id = uuid.uuid4().hex[:32]
        await client.post(
            f"https://aistudio.xiaomimimo.com/open-apis/chat/conversation/save?xiaomichatbot_ph={urllib.parse.quote(ph)}",
            headers=headers,
            cookies=cookies,
            json={"conversationId": conv_id, "title": "ASR", "type": "asr"}
        )

        rec_res = await client.post(
            f"https://aistudio.xiaomimimo.com/open-apis/asr/recognize?xiaomichatbot_ph={urllib.parse.quote(ph)}",
            headers=headers,
            cookies=cookies,
            json={
                "conversationId": conv_id,
                "msgId": msg_id,
                "audioUrl": resource_url,
                "language": language,
                "modelConfig": {"modelCode": "mimo-v2.5-asr"}
            }
        )
        rec_data = rec_res.json()
        if rec_data.get("code") != 0 or not rec_data.get("data"):
            raise RuntimeError(f"ASR recognize failed: {rec_data}")
        task_id = rec_data["data"].get("taskId")

        st_url = f"https://aistudio.xiaomimimo.com/open-apis/asr/recognizeStatus?xiaomichatbot_ph={urllib.parse.quote(ph)}&taskId={urllib.parse.quote(str(task_id))}"
        for _ in range(45):
            await asyncio.sleep(1.0)
            st_res = await client.get(st_url, headers=headers, cookies=cookies)
            if st_res.status_code == 200:
                sdata = st_res.json()
                if sdata.get("code") == 0 and sdata.get("data"):
                    st = sdata["data"].get("status")
                    if st == "success":
                        return sdata["data"].get("text", "")
                    elif st == "failed":
                        raise RuntimeError("ASR recognition failed upstream")

        raise TimeoutError("ASR recognition timed out")


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


def build_agent_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, framework: str = "default", is_ultraspeed: bool = False) -> str:
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
                clean_c = re.sub(r"\[Error from upstream: HTTP \d+[\s\S]*?\]", "", clean_c).strip()
                if clean_c:
                    formatted.append(f"Assistant: {clean_c}")
        return "\n\n".join(formatted)

    system_instructions = []
    history = []
    user_goals = []

    for idx, m in enumerate(messages):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            content = " ".join(text_parts)

        if role == "system":
            if content:
                clean_sys = sanitize_observation(content, max_chars=8000)
                system_instructions.append(clean_sys)
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
                actions = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "tool")
                    raw_args = fn.get("arguments", "")
                    if isinstance(raw_args, str) and raw_args.strip():
                        try:
                            parsed_args = json.loads(raw_args)
                            if isinstance(parsed_args, dict):
                                arg_str = ", ".join(f"{k}={v!r}" for k, v in list(parsed_args.items())[:3])
                                actions.append(f"{fn_name}({arg_str})")
                            else:
                                actions.append(f"{fn_name}()")
                        except Exception:
                            actions.append(f"{fn_name}()")
                    elif isinstance(raw_args, dict):
                        arg_str = ", ".join(f"{k}={v!r}" for k, v in list(raw_args.items())[:3])
                        actions.append(f"{fn_name}({arg_str})")
                    else:
                        actions.append(f"{fn_name}()")
                history.append(f"[Assistant Action]: Menjalankan tool: {', '.join(actions)}")
            elif content:
                clean_ast = re.sub(r"\[Error from upstream: HTTP \d+[\s\S]*?\]", "", content).strip()
                if clean_ast:
                    history.append(f"[Assistant]: {clean_ast}")
        elif role == "tool":
            tool_name = m.get("name", "tool")
            # Beri kuota lebih besar (sampai 8000 chars) untuk tool terbaru agar file/kode tidak terpotong
            obs_limit = 8000 if idx >= len(messages) - 2 else 2000
            sanitized = sanitize_observation(content, max_chars=obs_limit)
            # Tool results routinely contain JSON, which is full of double quotes.
            # Embedding that verbatim into this plain-text prompt leaves an
            # unescaped `{"` sequence that the model cannot parse, so it retries
            # the same tool until it burns its turn budget. Neutralise the quotes
            # so the observation reads as plain text.
            sanitized = sanitized.replace('"', "'")
            history.append(f"[Hasil {tool_name}]: {sanitized}")

    main_goal = user_goals[-1] if user_goals else ""

    MAX_PROMPT_BUDGET = 26000

    prompt_lines = []
    if system_instructions:
        sys_text = "\n\n".join(system_instructions)
        if len(sys_text) > 5000:
            sys_text = sys_text[:5000] + "\n...[instruksi sistem disingkat]"
        prompt_lines.append("Instruksi Sistem:\n" + sys_text)

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

    # Goal & status guidance
    goal_block = ""
    if main_goal:
        if messages and messages[-1].get("role") == "tool":
            goal_block = (
                f"Tugas utama user: {main_goal}\n"
                f"Status terkini: Hasil eksekusi tool terbaru ada di riwayat di atas.\n"
                f"- Jika tugas utama sudah terjawab/selesai secara lengkap, berikan jawaban akhir yang jelas dan informatif kepada user sekarang.\n"
                f"- Jika tugas masih membutuhkan langkah berikutnya atau tool lanjutan (misal melihat isi skill, memanggil MCP, membaca file, delegasi task), panggil tool berikutnya sekarang."
            )
        else:
            goal_block = f"Tugas sekarang:\n{main_goal}"

    # Calculate remaining budget for history
    fixed_len = sum(len(line) for line in prompt_lines) + len(goal_block) + 500
    hist_budget = max(4000, MAX_PROMPT_BUDGET - fixed_len)

    # Budget history turns: keep as many recent turns as fit in hist_budget
    if history:
        recent_hist = []
        current_hist_len = 0
        hist_window = 6 if is_ultraspeed else 16
        for item in reversed(history[-hist_window:]):
            item_len = len(item) + 1
            if current_hist_len + item_len <= hist_budget or not recent_hist:
                recent_hist.append(item)
                current_hist_len += item_len
            else:
                break
        recent_hist.reverse()
        prompt_lines.append("Riwayat percakapan:\n" + "\n".join(recent_hist))

    if goal_block:
        prompt_lines.append(goal_block)

    assembled = "\n\n".join(prompt_lines)
    if len(assembled) > MAX_PROMPT_BUDGET:
        assembled = assembled[:MAX_PROMPT_BUDGET]
    return assembled


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


TOOL_PARAM_ALIASES = {
    # File tools (read_file, write_file, patch)
    "path": ["file", "filename", "filepath", "target", "path_to_file", "file_path", "target_file", "path_name"],
    "content": ["text", "body", "data", "file_content", "contents", "code_to_write", "source_code", "code"],
    "code": ["script", "python", "command", "cmd", "code_snippet", "input", "py", "source"],
    "pattern": ["query", "regex", "search", "keyword", "term", "filter", "search_term"],
    "old_string": ["old_code", "find", "search", "original", "old", "target_string", "before", "match"],
    "new_string": ["new_code", "replace", "replacement", "updated", "new", "replacement_string", "after"],
    "command": ["cmd", "script", "shell_command", "bash", "sh"],
    # Discovery & introspection
    "queries": ["query", "searches", "search_queries", "keywords"],
    "names": ["name", "tool_names", "tools"],
    "calls": ["call", "invocations", "tool_calls"],
    # Web & Browser tools (Playwright MCP)
    "url": ["link", "href", "address", "target_url", "uri"],
    "selector": ["element", "target", "locator", "css", "xpath", "query", "target_element"],
    "expression": ["code", "script", "js", "function", "expr", "eval_code"],
}


def coerce_tool_args(name: str, args: dict, available_tools: Optional[List[Dict[str, Any]]] = None) -> dict:
    """Repair tool arguments whose JSON type or field names drifted from the declared schema."""
    if not isinstance(args, dict):
        return args

    snapshot = repr(args)  # cheap before-snapshot for coercion detection
    schema = _schema_for(name, available_tools) if available_tools else None
    if not isinstance(schema, dict):
        return args

    props = schema.get("properties") or {}
    required = schema.get("required") or []

    # 1. Alias healing for missing required and property fields
    for target_key in set(required) | set(props.keys()):
        if target_key not in args:
            alt_candidates = TOOL_PARAM_ALIASES.get(target_key, [])
            for alt in alt_candidates:
                if alt in args and alt not in props:
                    args[target_key] = args.pop(alt)
                    break

    # 2. Single-required property fallback:
    # If the schema requires exactly 1 field and args has only 1 field not matching the schema,
    # map that lone argument to the required field.
    if len(required) == 1:
        req_field = required[0]
        if req_field not in args and len(args) == 1:
            lone_key = list(args.keys())[0]
            if lone_key not in props:
                args[req_field] = args.pop(lone_key)

    # 3. Type coercion against schema properties
    for key, spec in props.items():
        if key in args and isinstance(spec, dict):
            args[key] = _coerce_value(args[key], spec)

    # 4. If a field requires array of objects (like tool_call's `calls`), ensure it is a list
    # and if caller provided a single dict, wrap it.
    for key, spec in props.items():
        if key in args and isinstance(spec, dict):
            types = _type_list(spec)
            if "array" in types and isinstance(args[key], dict):
                args[key] = [args[key]]

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

    # Auto-repair for write_file where model put top-level JSON fields directly into args (A11)
    if name == "write_file" and "content" not in args:
        if "path" in args:
            other_keys = {k: v for k, v in args.items() if k != "path"}
            if other_keys:
                args["content"] = json.dumps(other_keys, indent=2)
                for k in list(other_keys.keys()):
                    args.pop(k, None)

    # Auto-repair for clarify tool arguments
    if name == "clarify":
        if "questions" not in args:
            if "question" in args:
                q_val = args.pop("question")
                if isinstance(q_val, list):
                    args["questions"] = [{"question": str(item)} if not isinstance(item, dict) else item for item in q_val]
                elif isinstance(q_val, dict):
                    args["questions"] = [q_val]
                elif isinstance(q_val, str):
                    entry = {"question": q_val}
                    if "choices" in args and isinstance(args["choices"], list):
                        entry["choices"] = args.pop("choices")
                    args["questions"] = [entry]
            elif "prompt" in args:
                args["questions"] = [{"question": str(args.pop("prompt"))}]
            elif "text" in args:
                args["questions"] = [{"question": str(args.pop("text"))}]
        elif isinstance(args["questions"], list):
            args["questions"] = [
                {"question": str(item)} if not isinstance(item, dict) else item
                for item in args["questions"]
            ]
        elif isinstance(args["questions"], str):
            args["questions"] = [{"question": args["questions"]}]

    # Path unwrapping if provided as a single-element list
    if "path" in args and isinstance(args["path"], list) and len(args["path"]) == 1:
        args["path"] = str(args["path"][0])

    if name == "search_files":
        if "pattern" not in args:
            if "query" in args:
                args["pattern"] = args.pop("query")
            else:
                args["pattern"] = ""

    # Path normalization for file tools (generic & portable)
    if "path" in args and isinstance(args["path"], str):
        p = args["path"].replace("\\", "/").strip()
        workspace_root = os.environ.get("MIMOLY_WORKSPACE", "").replace("\\", "/").rstrip("/")
        if workspace_root:
            if p.startswith(workspace_root + "/"):
                pass  # already anchored to workspace
            elif not os.path.isabs(p) and not (len(p) > 2 and p[1] == ":"):
                # Anchor relative path to configured workspace
                p = f"{workspace_root}/{p}"
        args["path"] = p

    # Repair JSON-type drift (stringified arrays/objects) against the tool schema.
    args = coerce_tool_args(name, args, available_tools)

    # Surface (don't swallow) args that still don't fit the schema.
    _validation = validate_tool_args(name, args, available_tools)
    if not _validation["valid"]:
        _STATS["invalid_args"] += 1
        print(f"[mimoly] schema mismatch for '{name}': {_validation['issues']}")

    return args


def _recover_empty_execute_code(calls: List[Dict[str, Any]], reply_text: str) -> None:
    """Recover missing/empty 'code' argument in execute_code from surrounding markdown python blocks (A12)."""
    for tc in calls:
        if tc.get("name") == "execute_code":
            args = tc.setdefault("arguments", {})
            if not args.get("code"):
                m_py = re.findall(r"```(?:python|py)\n([\s\S]*?)\n```", reply_text)
                if m_py:
                    args["code"] = m_py[-1].strip()


def smart_extract_tool_calls(reply_text: str, user_prompt: str, available_tools: Optional[List[Dict[str, Any]]] = None):
    """Detect explicit JSON tool_calls or extract code blocks mapped to write/shell tools."""
    tool_names = [t.get("function", {}).get("name") for t in available_tools] if available_tools else []

    # 1. Cek explicit JSON or XML tool_calls
    # Pola 1A: XML-style <function=name> / <invoke name="name"> / <function name="name">
    xml_matches = re.findall(
        r"<(?:function|invoke)(?:=|\s+name=)[\"']?([a-zA-Z0-9_\-]+)[\"']?>([\s\S]*?)(?:</(?:function|invoke)>|$)",
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
                    r"<(?:parameter|param)(?:=|\s+name=)[\"']?([a-zA-Z0-9_\-]+)[\"']?>([\s\S]*?)(?:</(?:parameter|param)>|$)",
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
            _recover_empty_execute_code(tc_list, reply_text)
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
            clean_block = re.sub(r"^```(?:json)?\s*", "", clean_block)
            clean_block = re.sub(r"\s*```$", "", clean_block).strip()
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
            _recover_empty_execute_code(tc_list, reply_text)
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


@app.on_event("startup")
async def on_startup():
    try:
        load_account_pool()
        pool = get_account_pool()
        n = len(pool.accounts) if pool else 0
        print(f"[mimoly] Account pool ready: {n} credential(s); "
              f"admission_limit={_MIMOLY_ADMISSION_LIMIT or 'unlimited'}; "
              f"cache={'on' if _MIMOLY_CACHE_TTL > 0 else 'off'}")
    except Exception as e:
        print(f"[mimoly] Warning: account pool init failed: {e}")
    try:
        asyncio.create_task(check_cookie_health())
    except Exception:
        pass


@app.on_event("shutdown")
async def on_shutdown():
    try:
        await close_shared_client()
    except Exception:
        pass


@app.get("/")
@app.get("/health")
async def health():
    cookie_status = _COOKIE_HEALTH.get("status", "unknown")
    pool = get_account_pool()
    pool_snap = pool.snapshot() if pool else []
    return {
        "status": "ok",
        "service": "mimoly-web2api",
        "version": "3.0.0",
        "auth_configured": SESSION_FILE.exists(),
        "cookie_health": cookie_status,
        "accounts": {
            "total": len(pool_snap),
            "healthy": sum(1 for a in pool_snap if a.get("healthy")),
        },
        "admission": {
            "limit": _MIMOLY_ADMISSION_LIMIT or None,
            "in_flight": _ADMISSION.in_flight,
            "rejected": _ADMISSION.rejected,
        },
        "cache": _RESPONSE_CACHE.stats(),
        "rate_limit": _RATE_LIMITER.stats(),
    }


@app.get("/v1/stats")
async def stats():
    top_tools = sorted(_STATS["tool_usage"].items(), key=lambda x: -x[1])

    def _lat_summary(bucket):
        b = _STATS[bucket]
        n = b["count"]
        return {
            "count": n,
            "avg_ms": round(b["sum"] / n, 1) if n else 0.0,
            "min_ms": round(b["min"], 1) if b["min"] is not None else 0.0,
            "max_ms": round(b["max"], 1) if b["max"] is not None else 0.0,
        }

    # Timeline -> sorted list for the line chart (last 60 minutes)
    tl = _STATS["timeline"]
    timeline = [
        {"minute": m, **tl[m]}
        for m in sorted(tl.keys())[-60:]
    ]

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
        "finish_reasons": dict(_STATS["finish_reasons"]),
        "latency": _lat_summary("latency_ms"),
        "ttft": _lat_summary("ttft_ms"),
        "timeline": timeline,
        "tool_usage": dict(top_tools[:20]),
        "model_usage": _STATS["model_usage"],
        "model_tokens": _STATS["model_tokens"],
        "recent_requests": list(_REQUEST_LOG)[::-1][:50],
        "reliability": {
            "cache": {**_RESPONSE_CACHE.stats(), "hits_total": _STATS.get("cache_hits", 0)},
            "admission": {
                "limit": _MIMOLY_ADMISSION_LIMIT or None,
                "in_flight": _ADMISSION.in_flight,
                "accepted": _ADMISSION.accepted,
                "rejected": _ADMISSION.rejected,
            },
            "rate_limit": _RATE_LIMITER.stats(),
            "retries_total": _STATS.get("retries", 0),
            "rate_limited_total": _STATS.get("rate_limited", 0),
            "accounts": (get_account_pool().snapshot() if get_account_pool() else []),
        },
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
<title>mimoly · dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
// Fallback CDN: if jsdelivr failed, try a second mirror before charts init.
if (typeof Chart === 'undefined') {
  document.write('<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"><\/script>');
}
</script>
<style>
  :root {
    --bg:#FDF6E3; --ink:#0A0A0A; --paper:#FFFFFF;
    --yellow:#FFD23F; --cyan:#3DDBD9; --pink:#FF6BAA; --orange:#FF8A3D;
    --green:#7BE495; --blue:#6FA8FF; --purple:#B48CFF; --red:#FF5C5C;
    --bd:3px solid var(--ink); --sh:6px 6px 0 var(--ink);
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    font-family:'Segoe UI', system-ui, -apple-system, Arial, sans-serif;
    background:var(--bg); color:var(--ink);
    padding:26px 22px 60px;
    background-image:radial-gradient(rgba(10,10,10,.10) 1.4px, transparent 1.4px);
    background-size:22px 22px;
    -webkit-font-smoothing:antialiased;
  }
  .wrap { max-width:1180px; margin:0 auto; }

  /* ── Top bar ── */
  .topbar {
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    background:var(--yellow); border:var(--bd); box-shadow:var(--sh);
    padding:16px 20px; margin-bottom:26px;
  }
  .topbar h1 {
    font-size:30px; font-weight:900; letter-spacing:-1px; line-height:1;
  }
  .topbar .sub { font-weight:700; font-size:13px; opacity:.72; }
  .spacer { flex:1; }
  .pill {
    display:inline-flex; align-items:center; gap:8px;
    font-weight:900; font-size:12px; letter-spacing:.5px; text-transform:uppercase;
    background:var(--paper); border:var(--bd); padding:7px 14px;
  }
  .pill .dot { width:10px; height:10px; background:var(--red); border:2px solid var(--ink); }
  .pill.live .dot { background:var(--green); animation:blink 1.4s steps(1) infinite; }
  @keyframes blink { 50% { opacity:.25; } }
  .clock { font-weight:900; font-size:13px; }

  /* ── Section label ── */
  .label {
    display:inline-block; font-weight:900; font-size:13px; letter-spacing:1.2px;
    text-transform:uppercase; background:var(--ink); color:var(--paper);
    padding:6px 14px; margin:4px 0 16px;
  }

  /* ── Grids ── */
  .grid { display:grid; gap:18px; }
  .stat-grid { grid-template-columns:repeat(5,1fr); margin-bottom:32px; }
  .chart-grid { grid-template-columns:repeat(2,1fr); margin-bottom:32px; }
  .wide { grid-column:1 / -1; }
  @media (max-width:1000px){ .stat-grid{grid-template-columns:repeat(3,1fr);} }
  @media (max-width:760px){ .stat-grid{grid-template-columns:repeat(2,1fr);} .chart-grid{grid-template-columns:1fr;} }
  @media (max-width:440px){ .stat-grid{grid-template-columns:1fr;} }

  /* ── Stat card ── */
  .stat {
    background:var(--paper); border:var(--bd); box-shadow:var(--sh);
    padding:16px 16px 14px; position:relative; overflow:hidden;
  }
  .stat .k {
    font-weight:900; font-size:11px; letter-spacing:1px; text-transform:uppercase;
    margin-bottom:8px; display:flex; align-items:center; gap:7px;
  }
  .stat .k::before { content:''; width:11px; height:11px; background:var(--accent,var(--ink)); border:2px solid var(--ink); }
  .stat .v {
    font-weight:900; font-size:32px; line-height:1; letter-spacing:-1px;
    font-variant-numeric:tabular-nums; word-break:break-all;
  }
  .stat .u { font-weight:800; font-size:12px; opacity:.6; margin-top:5px; }
  .stat.hot { background:var(--accent); }

  /* ── Panel (chart / table) ── */
  .panel {
    background:var(--paper); border:var(--bd); box-shadow:var(--sh);
    padding:18px 18px 16px;
  }
  .panel h3 {
    font-weight:900; font-size:15px; text-transform:uppercase; letter-spacing:.6px;
    margin-bottom:14px; display:flex; align-items:center; gap:9px;
  }
  .panel h3::before { content:''; width:14px; height:14px; background:var(--accent,var(--yellow)); border:2px solid var(--ink); }
  .canvas-box { position:relative; height:250px; }
  .canvas-box.tall { height:290px; }
  .empty { font-weight:800; opacity:.45; padding:38px 0; text-align:center; font-size:14px; }

  /* ── Table ── */
  table { width:100%; border-collapse:collapse; }
  th {
    text-align:left; font-weight:900; font-size:11px; letter-spacing:.8px;
    text-transform:uppercase; border-bottom:3px solid var(--ink); padding:0 0 9px;
  }
  th:last-child, td:last-child { text-align:right; }
  td { padding:11px 0; border-bottom:2px solid rgba(10,10,10,.14); font-weight:700; font-size:14px; }
  td:last-child { font-variant-numeric:tabular-nums; }
  tr:last-child td { border-bottom:none; }
  .track { height:9px; background:rgba(10,10,10,.10); border:2px solid var(--ink); margin-top:6px; }
  .track > i { display:block; height:100%; background:var(--accent,var(--blue)); }

  /* ── Request log ── */
  .rlog { display:flex; flex-direction:column; gap:12px; max-height:640px; overflow-y:auto; }
  .rlog .row {
    border:2px solid var(--ink); background:#fff; padding:11px 13px;
  }
  .rlog .row.err { background:#FFE9E9; }
  .rlog .meta {
    display:flex; flex-wrap:wrap; gap:8px; align-items:center;
    font-weight:900; font-size:11px; letter-spacing:.5px; text-transform:uppercase;
    margin-bottom:8px;
  }
  .rlog .chip {
    border:2px solid var(--ink); padding:2px 8px; background:var(--paper);
    font-weight:900; font-size:10px; letter-spacing:.5px;
  }
  .rlog .chip.ok  { background:var(--green); }
  .rlog .chip.tc  { background:var(--yellow); }
  .rlog .chip.bad { background:var(--red); color:#fff; }
  .rlog .io { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .rlog .io > div { border-left:4px solid var(--ink); padding-left:9px; min-width:0; }
  .rlog .io .lbl {
    font-weight:900; font-size:10px; letter-spacing:.8px; text-transform:uppercase;
    opacity:.6; margin-bottom:3px;
  }
  .rlog .io .txt {
    font-weight:700; font-size:12.5px; line-height:1.45; white-space:pre-wrap;
    word-break:break-word; max-height:96px; overflow-y:auto;
  }
  .rlog .io .in  { border-left-color:var(--blue); }
  .rlog .io .out { border-left-color:var(--green); }
  .rlog .foot2 { font-weight:800; font-size:11px; opacity:.65; margin-top:8px; }
  @media (max-width:760px){ .rlog .io { grid-template-columns:1fr; } }

  /* ── Footer ── */
  .foot {
    margin-top:34px; display:flex; justify-content:space-between; align-items:center;
    flex-wrap:wrap; gap:10px; font-weight:800; font-size:12px;
    background:var(--paper); border:var(--bd); box-shadow:var(--sh); padding:12px 18px;
  }
  .foot a { color:var(--ink); font-weight:900; }
  .note { font-weight:800; font-size:12px; background:var(--orange); border:var(--bd); padding:8px 12px; margin-bottom:18px; display:none; }
</style>
</head>
<body>
<div class="wrap">

  <div class="topbar">
    <h1>MIMOLY</h1>
    <span class="sub">web2api · live dashboard</span>
    <span class="spacer"></span>
    <span class="pill" id="status"><span class="dot"></span><span id="status-t">connecting</span></span>
    <span class="clock" id="clock">--:--:--</span>
  </div>

  <div class="note" id="offline-note">⚠ Charts unavailable (Chart.js CDN not reachable) — numbers below are still live.</div>

  <span class="label">At a glance</span>
  <div class="grid stat-grid" id="stats"></div>

  <span class="label">Traffic &amp; latency</span>
  <div class="grid stat-grid" id="latency"></div>

  <span class="label">Charts</span>
  <div class="grid chart-grid">
    <div class="panel" style="--accent:var(--yellow)">
      <h3>Finish reasons</h3>
      <div class="canvas-box"><canvas id="c-finish"></canvas></div>
    </div>
    <div class="panel" style="--accent:var(--blue)">
      <h3>Token split</h3>
      <div class="canvas-box"><canvas id="c-tokens"></canvas></div>
    </div>
    <div class="panel" style="--accent:var(--orange)">
      <h3>Tool usage</h3>
      <div id="tool-table"><div class="empty">No tool calls yet</div></div>
    </div>
    <div class="panel" style="--accent:var(--purple)">
      <h3>Tokens per model</h3>
      <div id="model-table"><div class="empty">No model data yet</div></div>
    </div>
    <div class="panel wide" style="--accent:var(--green)">
      <h3>Activity timeline (last 60 min)</h3>
      <div class="canvas-box tall"><canvas id="c-timeline"></canvas></div>
    </div>
    <div class="panel wide" style="--accent:var(--cyan)">
      <h3>Recent requests <span class="chip" id="rlog-count" style="margin-left:auto">0</span></h3>
      <div class="rlog" id="rlog"><div class="empty">No requests yet</div></div>
    </div>
  </div>

  <div class="foot">
    <span>mimoly web2api v3.0.0 · <a href="/v1/stats">raw json</a> · <a href="/health">health</a></span>
    <span id="foot-meta">waiting for first poll…</span>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
const NB = ['#FFD23F','#3DDBD9','#FF6BAA','#FF8A3D','#7BE495','#6FA8FF','#B48CFF','#FF5C5C'];
const fmt = n => (n == null ? '—' : Number(n).toLocaleString());

function statCard(accent, label, value, unit, hot) {
  return '<div class="stat' + (hot ? ' hot' : '') + '" style="--accent:var(' + accent + ')">' +
    '<div class="k">' + label + '</div>' +
    '<div class="v">' + value + '</div>' +
    (unit ? '<div class="u">' + unit + '</div>' : '') +
  '</div>';
}

function uptime(sec) {
  sec = Math.floor(sec || 0);
  const d = Math.floor(sec/86400), h = Math.floor(sec%86400/3600),
        m = Math.floor(sec%3600/60), s = sec%60;
  if (d) return d + 'd ' + h + 'h';
  if (h) return h + 'h ' + m + 'm';
  if (m) return m + 'm ' + s + 's';
  return s + 's';
}

function rowsTable(rows, nameLabel, countLabel, accent) {
  if (!rows.length) return '<div class="empty">No data yet</div>';
  const max = Math.max(...rows.map(r => r[1]), 1);
  let h = '<table><tr><th>' + nameLabel + '</th><th>' + countLabel + '</th></tr>';
  for (const [name, val] of rows) {
    h += '<tr><td>' + name + '<div class="track" style="--accent:var(' + accent + ')">' +
      '<i style="width:' + Math.max(3, val/max*100) + '%"></i></div></td>' +
      '<td>' + fmt(val) + '</td></tr>';
  }
  return h + '</table>';
}

// ── Chart.js global neobrutalism defaults ──
let charts = {};
const OK = typeof Chart !== 'undefined';
if (OK) {
  Chart.defaults.font.family = "'Segoe UI', system-ui, Arial, sans-serif";
  Chart.defaults.font.weight = '800';
  Chart.defaults.color = '#0A0A0A';
  Chart.defaults.borderColor = '#0A0A0A';
}

function mkDoughnut(id) {
  if (!OK) return null;
  return new Chart($(id), {
    type: 'doughnut',
    data: { labels: [], datasets: [{ data: [], backgroundColor: NB, borderColor: '#0A0A0A', borderWidth: 3 }] },
    options: {
      responsive: true, maintainAspectRatio: false, cutout: '56%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, boxHeight: 12, padding: 10, font: { weight: '800', size: 11 } } },
        tooltip: { backgroundColor: '#0A0A0A', padding: 10, titleFont: { weight: '800' }, bodyFont: { weight: '700' } }
      }
    }
  });
}
function mkBar(id, horizontal) {
  if (!OK) return null;
  return new Chart($(id), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: NB, borderColor: '#0A0A0A', borderWidth: 3 }] },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#0A0A0A', padding: 10 } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { weight: '800', size: 11 } }, border: { width: 3 } },
        y: { grid: { color: 'rgba(10,10,10,.10)' }, ticks: { font: { weight: '800', size: 11 } }, border: { width: 3 } }
      }
    }
  });
}
function mkLine(id) {
  if (!OK) return null;
  return new Chart($(id), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'requests', data: [], borderColor: '#0A0A0A', backgroundColor: '#FFD23F', borderWidth: 3, stepped: true, pointRadius: 4, pointStyle: 'rect', pointBackgroundColor: '#FFD23F', pointBorderColor: '#0A0A0A', pointBorderWidth: 2, yAxisID: 'y' },
      { label: 'tokens', data: [], borderColor: '#0A0A0A', backgroundColor: '#3DDBD9', borderWidth: 3, tension: 0, pointRadius: 4, pointStyle: 'circle', pointBackgroundColor: '#3DDBD9', pointBorderColor: '#0A0A0A', pointBorderWidth: 2, yAxisID: 'y1' }
    ]},
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
      layout: { padding: { right: 10, left: 2, top: 4 } },
      plugins: {
        legend: { position: 'top', align: 'end', labels: { usePointStyle: true, boxWidth: 12, boxHeight: 12, font: { weight: '800', size: 12 } } },
        tooltip: { backgroundColor: '#0A0A0A', padding: 10 }
      },
      scales: {
        x: { grid: { display: false }, ticks: { font: { weight: '800', size: 10 } }, border: { width: 3 } },
        y: { position: 'left', min: 0, grid: { color: 'rgba(10,10,10,.10)' }, grace: '12%', ticks: { font: { weight: '800', size: 10 }, precision: 0 }, border: { width: 3 } },
        y1: { position: 'right', min: 0, grid: { display: false }, grace: '12%', ticks: { font: { weight: '800', size: 10 } }, border: { width: 3 } }
      }
    }
  });
}

function initCharts() {
  if (!OK) { $('offline-note').style.display = 'block'; return; }
  charts.finish = mkDoughnut('c-finish');
  charts.tokens = mkDoughnut('c-tokens');
  charts.timeline = mkLine('c-timeline');
}

const esc = s => String(s == null ? '' : s)
  .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
const ms = v => (v == null ? '—' : Math.round(v) + 'ms');

function renderRequestLog(rows) {
  $('rlog-count').textContent = rows.length;
  if (!rows.length) { $('rlog').innerHTML = '<div class="empty">No requests yet</div>'; return; }
  let h = '';
  for (const r of rows) {
    const bad = r.finish_reason === 'error';
    const fr = r.finish_reason || 'pending';
    const chipCls = bad ? 'bad' : (fr === 'tool_calls' ? 'tc' : 'ok');
    const when = new Date(r.ts * 1000).toLocaleTimeString();
    const tools = (r.tools_called && r.tools_called.length)
      ? '<span class="chip tc">🔧 ' + esc(r.tools_called.join(', ')) + '</span>' : '';
    h += '<div class="row' + (bad ? ' err' : '') + '">' +
      '<div class="meta">' +
        '<span class="chip ' + chipCls + '">' + esc(fr) + '</span>' +
        '<span class="chip">' + esc(r.model) + '</span>' +
        '<span class="chip">' + (r.stream ? 'stream' : 'sync') + '</span>' +
        tools +
        '<span style="margin-left:auto;opacity:.6">' + when + '</span>' +
      '</div>' +
      '<div class="io">' +
        '<div class="in"><div class="lbl">▲ Input</div><div class="txt">' + (esc(r.input) || '<em>(empty)</em>') + '</div></div>' +
        '<div class="out"><div class="lbl">▼ Output</div><div class="txt">' + (esc(r.output) || '<em>(no text)</em>') + '</div></div>' +
      '</div>' +
      '<div class="foot2">' + ms(r.latency_ms) + (r.ttft_ms != null ? ' · ttft ' + ms(r.ttft_ms) : '') +
        ' · in ' + fmt(r.prompt_tokens) + ' / out ' + fmt(r.completion_tokens) +
        ' tok · ' + esc(r.id) + '</div>' +
    '</div>';
  }
  $('rlog').innerHTML = h;
}

let alive = false;
async function tick() {
  try {
    const s = await (await fetch('/v1/stats', { cache: 'no-store' })).json();
    if (!alive) { alive = true; $('status').className = 'pill live'; $('status-t').textContent = 'live'; }
    $('clock').textContent = new Date().toLocaleTimeString();

    const t = s.tokens || {};
    const l = s.latency || {}, tt = s.ttft || {};
    const rate = s.requests > 0 ? (s.errors / s.requests * 100) : 0;

    $('stats').innerHTML =
      statCard('--blue',   'Requests',    fmt(s.requests), 'total received') +
      statCard('--red',    'Errors',      fmt(s.errors),   'failed requests', s.errors > 0) +
      statCard('--orange', 'Error rate',  (Math.round(rate*10)/10) + '%', 'errors / requests') +
      statCard('--yellow', 'Tool calls',  fmt(s.tool_calls), 'invoked tools') +
      statCard('--cyan',   'Coerced',     fmt(s.coerced_args), 'args auto-repaired') +
      statCard('--pink',   'Invalid args', fmt(s.invalid_args), 'schema mismatches', s.invalid_args > 0) +
      statCard('--purple', 'Input tokens', fmt(t.prompt_tokens), 'prompt') +
      statCard('--green',  'Output tokens', fmt(t.completion_tokens), 'completion') +
      statCard('--blue',   'Total tokens', fmt(t.total_tokens), 'in + out') +
      statCard('--orange', 'Reasoning',   fmt(t.reasoning_tokens), 'thinking tokens');

    $('latency').innerHTML =
      statCard('--yellow', 'Avg latency', l.avg_ms ? l.avg_ms + ' ms' : '—', l.count + ' samples') +
      statCard('--cyan',   'Avg TTFT',    tt.avg_ms ? tt.avg_ms + ' ms' : '—', tt.count + (tt.count === 1 ? ' stream' : ' streams')) +
      statCard('--green',  'Min latency', l.count ? l.min_ms + ' ms' : '—', 'fastest') +
      statCard('--pink',   'Max latency', l.count ? l.max_ms + ' ms' : '—', 'slowest') +
      statCard('--purple', 'Uptime', uptime(s.uptime_seconds), s.requests + ' req served');

    // Doughnut: finish reasons
    const fr = s.finish_reasons || {};
    const frKeys = Object.keys(fr);
    const frTotal = frKeys.reduce((a, k) => a + fr[k], 0);
    const frLabels = frKeys.map(k => k + ': ' + fr[k] + (frTotal ? ' (' + Math.round(fr[k]/frTotal*100) + '%)' : ''));
    const frData = frKeys.map(k => fr[k]);
    if (charts.finish) {
      charts.finish.data.labels = frKeys.length ? frLabels : ['no data'];
      charts.finish.data.datasets[0].data = frKeys.length ? frData : [1];
      charts.finish.data.datasets[0].backgroundColor = frKeys.length ? NB : ['#e6e6e6'];
      charts.finish.update('none');
    }
    // Doughnut: token split
    const splitVals = [t.prompt_tokens||0, t.completion_tokens||0, t.reasoning_tokens||0];
    const splitNames = ['input', 'output', 'reasoning'];
    const splitTotal = splitVals.reduce((a, b) => a + b, 0);
    const hasSplit = splitVals.some(v => v > 0);
    if (charts.tokens) {
      charts.tokens.data.labels = hasSplit
        ? splitNames.map((n, i) => n + ': ' + fmt(splitVals[i]) + (splitTotal ? ' (' + Math.round(splitVals[i]/splitTotal*100) + '%)' : ''))
        : ['no data'];
      charts.tokens.data.datasets[0].data = hasSplit ? splitVals : [1];
      charts.tokens.data.datasets[0].backgroundColor = hasSplit ? [NB[5], NB[4], NB[2]] : ['#e6e6e6'];
      charts.tokens.update('none');
    }
    // Timeline
    const tl = s.timeline || [];
    if (charts.timeline) {
      charts.timeline.data.labels = tl.map(r => new Date(r.minute*1000).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'}));
      charts.timeline.data.datasets[0].data = tl.map(r => r.requests);
      charts.timeline.data.datasets[1].data = tl.map(r => r.total_tokens);
      charts.timeline.update('none');
    }

    $('tool-table').innerHTML = rowsTable(
      Object.entries(s.tool_usage || {}).sort((a,b) => b[1]-a[1]), 'Tool', 'Calls', '--orange');
    $('model-table').innerHTML = rowsTable(
      Object.entries(s.model_tokens || {}).map(([m,v]) => [m, v.total_tokens]).sort((a,b) => b[1]-a[1]),
      'Model', 'Tokens', '--purple');

    renderRequestLog(s.recent_requests || []);

    $('foot-meta').textContent = 'up ' + uptime(s.uptime_seconds) + ' · ' +
      fmt(s.requests) + ' req · ' + fmt(t.total_tokens) + ' tokens · updated ' + new Date().toLocaleTimeString();
  } catch (e) {
    alive = false;
    $('status').className = 'pill';
    $('status-t').textContent = 'offline';
    $('foot-meta').textContent = 'connection lost — retrying…';
  }
}

initCharts();
tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


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


async def ensure_new_conversation(save_url: str, headers: dict, cookies: dict, model_name: str) -> str:
    """Register a fresh conversation ID with Xiaomi upstream using shared client."""
    new_cid = uuid.uuid4().hex
    try:
        client = get_shared_client()
        await client.post(
            save_url,
            headers=headers,
            cookies=cookies,
            json={"conversationId": new_cid, "type": "chat", "title": f"Mimoly {model_name}"},
            timeout=8.0,
        )
    except Exception as e:
        print(f"[mimoly] Warning: conversation registration failed: {e}")
    return new_cid


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    return await handle_chat_completion(body, request)


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API endpoint (/v1/messages) for Claude Code, Cline, etc."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid JSON body"}}, status_code=400)

    openai_body = convert_anthropic_request(body)
    model = body.get("model", "mimo-v2.6-pro")
    stream = body.get("stream", False)

    resp = await handle_chat_completion(openai_body, request)
    if not stream:
        if isinstance(resp, dict):
            if "error" in resp:
                return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(resp["error"])}}, status_code=400)
            ant_resp = convert_openai_to_anthropic_response(resp, model)
            return JSONResponse(ant_resp)
        elif isinstance(resp, JSONResponse):
            try:
                data = json.loads(resp.body.decode("utf-8"))
                if "error" in data:
                    return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(data["error"])}}, status_code=resp.status_code)
                ant_resp = convert_openai_to_anthropic_response(data, model)
                return JSONResponse(ant_resp)
            except Exception:
                return resp
        return resp
    return resp


@app.post("/v1/responses")
async def openai_responses(request: Request):
    """OpenAI Responses API endpoint (/v1/responses)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    chat_body = convert_responses_request_to_chat(body)
    model = chat_body.get("model", "mimo-v2.6-pro")
    stream = chat_body.get("stream", False)

    resp = await handle_chat_completion(chat_body, request)
    if not stream:
        if isinstance(resp, dict):
            if "error" in resp:
                return JSONResponse({"error": str(resp["error"])}, status_code=400)
            return JSONResponse(convert_chat_to_responses_output(resp, model))
        elif isinstance(resp, JSONResponse):
            try:
                data = json.loads(resp.body.decode("utf-8"))
                if "error" in data:
                    return resp
                return JSONResponse(convert_chat_to_responses_output(data, model))
            except Exception:
                return resp
        return resp
    return resp


@app.post("/v1/audio/speech")
async def audio_speech(request: Request):
    """OpenAI-compatible Text-To-Speech endpoint (/v1/audio/speech)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    text = body.get("input", "")
    if not text:
        return JSONResponse({"error": "'input' text is required"}, status_code=400)

    voice = body.get("voice", "alloy")
    model = body.get("model", "mimo-v2.5-tts")

    try:
        audio_bytes = await tts_generate_audio(text, voice=voice, model=model)
        return Response(content=audio_bytes, media_type="audio/wav")
    except Exception as e:
        print(f"[mimoly] TTS error: {e}")
        return JSONResponse({"error": f"TTS generation failed: {e}"}, status_code=502)


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(request: Request):
    """OpenAI Whisper-compatible Speech-To-Text endpoint (/v1/audio/transcriptions)."""
    try:
        form = await request.form()
        file = form.get("file")
        if not file:
            return JSONResponse({"error": "Audio file 'file' is required"}, status_code=400)
        audio_bytes = await file.read()
        if not audio_bytes:
            return JSONResponse({"error": "Audio file is empty"}, status_code=400)

        filename = getattr(file, "filename", "audio.wav") or "audio.wav"
        language = str(form.get("language", "auto"))
        response_format = str(form.get("response_format", "json"))
        text = await asr_transcribe_audio(audio_bytes, filename=filename, language=language)

        if response_format == "text":
            return Response(content=text, media_type="text/plain")
        return JSONResponse({"text": text})
    except Exception as e:
        print(f"[mimoly] ASR error: {e}")
        return JSONResponse({"error": f"ASR transcription failed: {e}"}, status_code=502)


def _client_key(request: Request) -> str:
    """Best-effort client identity for rate limiting (honours common proxy headers)."""
    try:
        for hdr in ("x-forwarded-for", "x-real-ip", "cf-connecting-ip"):
            v = request.headers.get(hdr)
            if v:
                return v.split(",")[0].strip()
        if request.client and request.client.host:
            return request.client.host
    except Exception:
        pass
    return "unknown"


async def handle_chat_completion(body: dict, request: Request):
    """Public entrypoint: rate limit + admission control + exact cache wrapper.

    * Per-IP rate limit (opt-in via MIMOLY_RATE_LIMIT) returns 429 + Retry-After.
    * Admission control rejects fast (HTTP 503) when too many requests are
      already in flight, instead of letting an upstream slowdown queue forever.
    * Exact-match TTL cache serves identical *non-streaming* requests without
      touching upstream (streaming requests always bypass the cache).
    * The actual work lives in ``_handle_chat_completion_inner``.
    """
    # --- per-IP rate limit (cheap, before anything else) ---
    if _MIMOLY_RATE_LIMIT:
        _ck = _client_key(request)
        if not _RATE_LIMITER.allow(_ck):
            _STATS["errors"] += 1
            _STATS["rate_limited"] = _STATS.get("rate_limited", 0) + 1
            _retry = max(1, int(_RATE_LIMITER.retry_after(_ck) + 0.999))
            return JSONResponse(
                {"error": {
                    "message": f"Rate limit exceeded: max {_MIMOLY_RATE_LIMIT} requests per {_MIMOLY_RATE_WINDOW:g}s.",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }},
                status_code=429,
                headers={"Retry-After": str(_retry)},
            )

    _admitted = _ADMISSION.acquire()
    if not _admitted:
        _STATS["errors"] += 1
        return JSONResponse(
            {"error": {
                "message": "Server overloaded: too many in-flight requests. Please retry shortly.",
                "type": "rate_limit_error",
                "code": "overloaded",
            }},
            status_code=503,
            headers={"Retry-After": "2"},
        )

    stream = bool(body.get("stream"))
    cache_key = None
    if not stream:
        try:
            cache_key = _RESPONSE_CACHE.make_key(body)
            cached = _RESPONSE_CACHE.get(cache_key)
        except Exception:
            cached = None
        if cached is not None:
            _STATS["cache_hits"] = _STATS.get("cache_hits", 0) + 1
            _ADMISSION.release()
            return JSONResponse(cached)

    defer_release = False
    try:
        result = await _handle_chat_completion_inner(body, request)
        # Cache only successful, fully-formed non-streaming completions.
        if (cache_key is not None and isinstance(result, dict)
                and "choices" in result and "error" not in result):
            try:
                _RESPONSE_CACHE.set(cache_key, result)
            except Exception:
                pass
        # Surface retry activity to the caller (non-streaming; streaming already
        # started so headers are immutable by then).
        if isinstance(result, dict):
            try:
                result = JSONResponse(result, headers={"X-Retry-Attempt-Count": str(_RETRY_ATTEMPTS.get())})
            except Exception:
                pass
        if isinstance(result, StreamingResponse):
            defer_release = True
            _inner_iter = result.body_iterator

            async def _admitting_iter():
                try:
                    async for _chunk in _inner_iter:
                        yield _chunk
                finally:
                    _ADMISSION.release()

            result.body_iterator = _admitting_iter()
        return result
    finally:
        if not defer_release:
            _ADMISSION.release()


async def _handle_chat_completion_inner(body: dict, request: Request):

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
    _record_timeline(requests=1)
    _req_start = time.time()

    # Per-request log: create the record up front so even early failures are logged.
    _tool_names = [t.get("function", {}).get("name", "?") for t in (tools or []) if isinstance(t, dict)]
    _req_rec = _new_request(
        model=model, stream=bool(stream), messages=len(messages),
        tools=len(tools) if tools else 0, prompt=_last_user_text(messages),
        tools_names=_tool_names,
    )

    # Cap thinking + answer output via max_completion_tokens if not set by caller
    # (prevents unbounded thinking bloat - research shows MiMo can produce 25-30k thinking tokens without cap)
    if not body.get("max_completion_tokens") and not body.get("max_tokens"):
        body["max_completion_tokens"] = 8192

    try:
        with open("last_request.json", "w", encoding="utf-8") as f_req:
            json.dump(body, f_req, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # Load authentic session cookies (from the rotating account pool if available)
    _acct = None
    try:
        cookies, _acct = acquire_account()
    except Exception as e:
        _STATS["errors"] += 1
        _record_timeline(errors=1)
        _record_finish_reason("error")
        _finish_request(_req_rec, output=f"[error] {e}", finish_reason="error",
                        latency_ms=(time.time() - _req_start) * 1000.0)
        return JSONResponse({"error": str(e)}, status_code=401)

    # Fast path for background session title generation: Hermes CLI fires this in background on turn 1
    if not stream and len(messages) <= 2:
        sys_txt = str(messages[0].get("content", "")) if messages else ""
        if "name chat sessions" in sys_txt.lower() or "title that lets them find this conversation" in sys_txt.lower():
            user_txt = str(messages[-1].get("content", "")) if len(messages) > 1 else "Chat"
            words = [w for w in re.sub(r"[^\w\s-]", "", user_txt).split() if len(w) > 1][:5]
            title = " ".join(words).title() or "Diskusi Projek"
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created_time,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": title},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }

    # Format unified prompt (with per-framework tool template)
    if tools or len(messages) > 1:
        agent_framework = detect_agent_framework(request, body)
        print(f"[mimoly] agent_framework={agent_framework} (profile: {AGENT_PROFILES[agent_framework]['label']})")
        is_ultra = (target_upstream_model == "mimo-v2.5-pro-ultraspeed")
        user_prompt = build_agent_prompt(messages, tools, framework=agent_framework, is_ultraspeed=is_ultra)
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
    upstream_url, save_url = get_upstream_endpoints(target_upstream_model, ph_param)

    headers = {
        "Content-Type": "application/json",
        "Accept-Language": "system",
        "x-timeZone": "Asia/Bangkok",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }

    # Ensure conversation is registered in Xiaomi database
    conv_id = await ensure_new_conversation(save_url, headers, cookies, target_upstream_model)

    # Multimodal: Detect images and upload to Xiaomi OSS
    multi_medias = []
    if ph_param:
        try:
            multi_medias = await prepare_multimedias_for_request(
                messages, cookies, ph_param, model=target_upstream_model
            )
            if multi_medias:
                print(f"[mimoly] Prepared {len(multi_medias)} multimodal attachment(s) for upstream")
        except Exception as e:
            print(f"[mimoly] Warning: failed to prepare multimedias: {e}")

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
        "multiMedias": multi_medias
    }

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_time = int(time.time())

    # --- STREAMING PATH ---
    if stream:
        async def sse_generator():
            # Initialize state BEFORE the try so the finally safety-net can always
            # reference them, even if the client disconnects on the very first yield.
            # NOTE: use _cur_* locals — assigning to the closure names (upstream_url,
            # cookies, _acct) inside a nested function would shadow them and raise
            # UnboundLocalError on the read that happens before any assignment.
            _cur_cookies = cookies
            _cur_upstream_url = upstream_url
            _cur_acct = _acct
            _acct_released = False
            accumulated_chunks = []
            usage_data = None
            in_thinking = False
            _first_token_at = None
            _finish_reason = "stop"
            _stream_out_text = ""
            _stream_tools_called: List[str] = []
            _initial_chunk_sent = False
            _acct_lock = _cur_acct.get_lock() if (_cur_acct and hasattr(_cur_acct, "get_lock")) else None
            if _acct_lock:
                await _acct_lock.acquire()
            try:
                try:
                    _budget = RetryBudget(max_attempts=5, base_delay=1.0, max_delay=12.0, deadline=50.0, min_delay=0.2)
                    _budget.start()
                    _stream_started = False  # "started guard": no retry after first content byte
                    _content_started = False # True only when actual delta.content is yielded
                    _last_error_class = None
                    for attempt in range(5):
                        _upstream_was_busy = False
                        upstream_payload["msgId"] = uuid.uuid4().hex
                        try:
                            if _cur_acct and hasattr(_cur_acct, "wait_throttle"):
                                await _cur_acct.wait_throttle(min_interval=2.0)
                            _client = get_shared_client()
                            async with _client.stream("POST", _cur_upstream_url, headers=headers, cookies=_cur_cookies, json=upstream_payload) as resp:
                                    if resp.status_code != 200:
                                        err_text = await resp.aread()
                                        _err_class = classify_status(resp.status_code)
                                        _last_error_class = _err_class
                                        _retryable_now = is_retryable(_err_class) or (
                                            _err_class == ERROR_AUTH and can_rotate_account())
                                        if _retryable_now and _budget.should_retry(attempt) and not _stream_started and (tools or not accumulated_chunks):
                                            print(f"[mimoly] Upstream stream HTTP {resp.status_code} ({_err_class}), retrying attempt {attempt+1}/5...")
                                            _bump_retry()
                                            # Rotate to another credential when the failure is key-specific AND multiple accounts exist.
                                            if (should_rotate_key(_err_class) or _err_class == ERROR_AUTH) and can_rotate_account():
                                                release_account(_cur_acct, success=False, error_class=_err_class)
                                                try:
                                                    _cur_cookies, _cur_acct = acquire_account()
                                                    _ph = _cur_cookies.get("xiaomichatbot_ph", "")
                                                    _cur_upstream_url, _ = get_upstream_endpoints(target_upstream_model, _ph)
                                                except Exception:
                                                    pass
                                            await _budget.sleep(attempt, error_class=_err_class)
                                            continue
                                        if resp.status_code in (401, 403):
                                            print(f"[mimoly] Upstream auth failed HTTP {resp.status_code}. Session cookies in session.json may be expired!")
                                            err_detail = "Unauthorized/Forbidden - please refresh cookies in session.json"
                                        else:
                                            err_detail = err_text.decode('utf-8', errors='ignore')
                                        err_chunk = {
                                            "id": completion_id,
                                            "object": "chat.completion.chunk",
                                            "created": created_time,
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": f"\n[Error from upstream: HTTP {resp.status_code} ({err_detail})]"},
                                                "finish_reason": "error"
                                            }]
                                        }
                                        yield f"data: {json.dumps(err_chunk)}\n\n"
                                        _STATS["errors"] += 1
                                        _record_timeline(errors=1)
                                        _record_finish_reason("error")
                                        _finish_request(_req_rec, output=f"[error] upstream HTTP {resp.status_code}",
                                                        finish_reason="error",
                                                        latency_ms=(time.time() - _req_start) * 1000.0)
                                        release_account(_cur_acct, success=False, error_class=_err_class)
                                        _acct_released = True
                                        yield "data: [DONE]\n\n"
                                        return

                                    _stream_started = True
                                    if not tools and not _initial_chunk_sent:
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
                                        _initial_chunk_sent = True

                                    last_event = None
                                    line_iter = resp.aiter_lines().__aiter__()
                                    _disc_checked = 0.0
                                    _upstream_was_busy = False
                                    while True:
                                        try:
                                            line = await asyncio.wait_for(line_iter.__anext__(), timeout=10.0)
                                        except asyncio.TimeoutError:
                                            # Send SSE comment ping to prevent reverse proxies/clients timing out during deep reasoning
                                            yield ": keep-alive\n\n"
                                            # Also poll for client disconnect during long idle gaps
                                            # (e.g. deep thinking) so we can cancel upstream work.
                                            _now = time.time()
                                            if _now - _disc_checked >= 5.0:
                                                _disc_checked = _now
                                                try:
                                                    if await request.is_disconnected():
                                                        print("[mimoly] Client disconnected during stream; cancelling upstream.")
                                                        break
                                                except Exception:
                                                    pass
                                            continue
                                        except StopAsyncIteration:
                                            break

                                        if not line:
                                            continue
                                        if line.startswith("event:"):
                                            last_event = line[6:].strip()
                                            continue
                                        if not line.startswith("data:"):
                                            continue

                                        data_str = line[5:].strip()
                                        if "[DONE]" in data_str:
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
                                        # Ignore dialog ID event (both standard cluster and FastChat 6-digit IDs)
                                        if last_event == "dialogId" or (parsed.get("type") is None and str(content_piece).isdigit()) or (str(content_piece).isdigit() and len(str(content_piece)) >= 5):
                                            continue

                                        if not content_piece:
                                            continue

                                        # Detect upstream rate-limit or busy messages before content streaming begins
                                        if (not _content_started or tools) and is_upstream_busy(content_piece):
                                            print(f"[mimoly] Upstream emitted busy phrase in frame: {content_piece.strip()!r}")
                                            _upstream_was_busy = True
                                            break

                                        # Time-to-first-token (stream): first real content chunk
                                        if _first_token_at is None:
                                            _first_token_at = time.time()
                                            _stream_started = True
                                            _record_latency("ttft_ms", (_first_token_at - _req_start) * 1000.0)

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
                                                    _content_started = True
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
                                                _content_started = True
                                                accumulated_chunks.append(clean_piece)
                            if (_upstream_was_busy or is_upstream_busy("".join(accumulated_chunks))) and attempt < 4:
                                print(f"[mimoly] Upstream stream busy, retrying attempt {attempt+1}/5...")
                                accumulated_chunks.clear()
                                _bump_retry()
                                new_cid = await ensure_new_conversation(save_url, headers, _cur_cookies, target_upstream_model)
                                upstream_payload["conversationId"] = new_cid
                                upstream_payload["msgId"] = uuid.uuid4().hex
                                await asyncio.sleep(2.5 * (attempt + 1))
                                continue
                            break
                        except Exception as conn_err:
                            _conn_class = classify_exception(conn_err)
                            if not _budget.should_retry(attempt) or _stream_started:
                                raise conn_err
                            print(f"[mimoly] Upstream stream retry {attempt+1}/3 due to: {conn_err}")
                            _bump_retry()
                            await _budget.sleep(attempt)

                    # Reached here with content (or after final attempt): account served OK.
                    release_account(_cur_acct, success=True)
                    _acct_released = True

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
                    _STATS["errors"] += 1
                    _record_timeline(errors=1)
                    _record_finish_reason("error")
                    _finish_request(_req_rec, output=f"[error] stream: {e}", finish_reason="error",
                                    latency_ms=(time.time() - _req_start) * 1000.0)
                    yield "data: [DONE]\n\n"
                    return

                full_reply = "".join(accumulated_chunks)

                # If tools were active, parse tools and stream tool_calls
                _stream_out_text = full_reply
                _stream_tools_called: List[str] = []
                if tools:
                    raw_calls, clean_text = smart_extract_tool_calls(full_reply, user_prompt, tools)
                    _stream_out_text = clean_text or full_reply
                    _stream_tools_called = [tc.get("name", "?") for tc in raw_calls]
                    if raw_calls:
                        _STATS["tool_calls"] += len(raw_calls)
                        _record_timeline(tool_calls=len(raw_calls))
                        _finish_reason = "tool_calls"
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
                        if not clean_text:
                            clean_text = "Langkah telah selesai diproses."
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
                    if not _content_started:
                        # Model produced thinking tokens but zero trailing text
                        fallback_msg = "Langkah telah selesai diproses."
                        chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created_time,
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": fallback_msg},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        _content_started = True
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

                _record_latency("latency_ms", (time.time() - _req_start) * 1000.0)
                _record_finish_reason(_finish_reason)
                _finish_request(
                    _req_rec,
                    output=_stream_out_text,
                    finish_reason=_finish_reason,
                    latency_ms=(time.time() - _req_start) * 1000.0,
                    ttft_ms=((_first_token_at - _req_start) * 1000.0) if _first_token_at else None,
                    usage=usage_data,
                    tools_called=_stream_tools_called,
                )
                yield "data: [DONE]\n\n"
            finally:
                if _acct_lock and hasattr(_acct_lock, "locked") and _acct_lock.locked():
                    try:
                        _acct_lock.release()
                    except RuntimeError:
                        pass
                # Safety net: if the client disconnected or the stream was
                # cancelled before the normal finalize ran, record what we have
                # so a request never stays "pending" forever, and always return
                # the pooled account (release is idempotent per account).
                if not _acct_released:
                    # A client-initiated abort is not the account's fault: return it
                    # as a success so frequent cancels can never trip its breaker.
                    release_account(_cur_acct, success=True)
                    _acct_released = True
                if _req_rec is not None and not _req_rec.get("finish_reason"):
                    _elapsed_ms = (time.time() - _req_start) * 1000.0
                    _record_latency("latency_ms", _elapsed_ms)
                    _record_finish_reason("cancelled")
                    _finish_request(
                        _req_rec,
                        output=_stream_out_text or "".join(accumulated_chunks),
                        finish_reason="cancelled",
                        latency_ms=_elapsed_ms,
                        ttft_ms=((_first_token_at - _req_start) * 1000.0) if _first_token_at else None,
                        usage=usage_data,
                        tools_called=_stream_tools_called,
                    )

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

    _acct_lock = _acct.get_lock() if (_acct and hasattr(_acct, "get_lock")) else None
    if _acct_lock:
        await _acct_lock.acquire()
    try:
        _budget = RetryBudget(max_attempts=5, base_delay=1.0, max_delay=12.0, deadline=50.0, min_delay=0.2)
        _budget.start()
        for attempt in range(5):
            upstream_payload["msgId"] = uuid.uuid4().hex
            try:
                if _acct and hasattr(_acct, "wait_throttle"):
                    await _acct.wait_throttle(min_interval=2.0)
                _client = get_shared_client()
                async with _client.stream("POST", upstream_url, headers=headers, cookies=cookies, json=upstream_payload) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            _err_class = classify_status(resp.status_code)
                            _retryable_now = is_retryable(_err_class) or (
                                _err_class == ERROR_AUTH and can_rotate_account())
                            if _retryable_now and _budget.should_retry(attempt):
                                print(f"[mimoly] Upstream HTTP {resp.status_code} ({_err_class}), retrying attempt {attempt+1}/4...")
                                _bump_retry()
                                if (should_rotate_key(_err_class) or _err_class == ERROR_AUTH) and can_rotate_account():
                                    release_account(_acct, success=False, error_class=_err_class)
                                    try:
                                        cookies, _acct = acquire_account()
                                        ph_param = cookies.get("xiaomichatbot_ph", "")
                                        upstream_url, _ = get_upstream_endpoints(target_upstream_model, ph_param)
                                    except Exception:
                                        pass
                                await _budget.sleep(attempt, error_class=_err_class)
                                continue
                            if resp.status_code in (401, 403):
                                print(f"[mimoly] Upstream auth failed HTTP {resp.status_code}. Session cookies in session.json may be expired!")
                                err_detail = "Unauthorized/Forbidden - please refresh cookies in session.json"
                            else:
                                err_detail = err_body.decode('utf-8', errors='ignore')
                            _STATS["errors"] += 1
                            _record_timeline(errors=1)
                            _record_finish_reason("error")
                            _finish_request(_req_rec, output=f"[error] upstream HTTP {resp.status_code}",
                                            finish_reason="error",
                                            latency_ms=(time.time() - _req_start) * 1000.0)
                            release_account(_acct, success=False, error_class=_err_class)
                            return JSONResponse(
                                {"error": f"Upstream HTTP {resp.status_code} ({err_detail})"},
                                status_code=resp.status_code
                            )

                        last_event = None
                        async for line in resp.aiter_lines():
                            if line.startswith("event:"):
                                last_event = line[6:].strip()
                                continue
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
                            if last_event == "dialogId" or (parsed.get("type") is None and str(content_piece).isdigit()) or (str(content_piece).isdigit() and len(str(content_piece)) >= 5):
                                continue
                            if content_piece:
                                accumulated_chunks.append(content_piece.replace("\x00", ""))

                if is_upstream_busy("".join(accumulated_chunks)) and attempt < 4:
                    print(f"[mimoly] Upstream server busy, retrying attempt {attempt+1}/5...")
                    accumulated_chunks.clear()
                    _bump_retry()
                    new_cid = await ensure_new_conversation(save_url, headers, cookies, target_upstream_model)
                    upstream_payload["conversationId"] = new_cid
                    upstream_payload["msgId"] = uuid.uuid4().hex
                    await asyncio.sleep(2.5 * (attempt + 1))
                    continue
                break
            except Exception as conn_err:
                _conn_class = classify_exception(conn_err)
                if not _budget.should_retry(attempt):
                    raise conn_err
                print(f"[mimoly] Upstream non-stream retry {attempt+1}/3 due to: {conn_err}")
                _bump_retry()
                await _budget.sleep(attempt)

        # Reached here with a complete answer: the account served us fine.
        release_account(_acct, success=True)

    except Exception as e:
        _STATS["errors"] += 1
        _record_timeline(errors=1)
        _record_finish_reason("error")
        _finish_request(_req_rec, output=f"[error] upstream: {e}", finish_reason="error",
                        latency_ms=(time.time() - _req_start) * 1000.0)
        release_account(_acct, success=False, error_class=classify_exception(e))
        return JSONResponse({"error": f"Failed to connect to upstream: {e}"}, status_code=502)
    finally:
        if _acct_lock and hasattr(_acct_lock, "locked") and _acct_lock.locked():
            try:
                _acct_lock.release()
            except RuntimeError:
                pass

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
        _record_timeline(tool_calls=len(raw_calls))
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

    _record_latency("latency_ms", (time.time() - _req_start) * 1000.0)
    _record_finish_reason(finish_reason)

    _finish_request(
        _req_rec,
        output=clean_text or full_reply or (reasoning_text or ""),
        finish_reason=finish_reason,
        latency_ms=(time.time() - _req_start) * 1000.0,
        usage=usage_data,
        tools_called=[tc.get("name", "?") for tc in raw_calls],
    )

    if not openai_tool_calls and not clean_text and not full_reply:
        if reasoning_text:
            clean_text = "Langkah telah selesai diproses."

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
    global SESSION_FILE
    if getattr(args, "session_file", None):
        SESSION_FILE = Path(args.session_file).resolve()
    if getattr(args, "workspace", None):
        os.environ["MIMOLY_WORKSPACE"] = args.workspace
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
    global SESSION_FILE
    if getattr(args, "session_file", None):
        SESSION_FILE = Path(args.session_file).resolve()
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
                last_event = None
                async for line in resp.aiter_lines():
                    if line.startswith("event:"):
                        last_event = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            d = json.loads(data_str)
                            c = d.get("content", "")
                            if last_event == "dialogId" or (d.get("type") is None and str(c).isdigit()) or (str(c).isdigit() and len(str(c)) >= 5):
                                continue
                            if c:
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
    global SESSION_FILE
    if getattr(args, "session_file", None):
        SESSION_FILE = Path(args.session_file).resolve()
    import websockets

    def find_system_browser() -> str:
        candidates = [
            # Windows Chrome
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            # Windows Edge
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Edge\Application\msedge.exe"),
            # macOS Chrome & Edge
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
            # Linux & PATH
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "msedge",
            "microsoft-edge",
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
    serve_parser.add_argument("--session-file", default=None, help="Custom path to session.json")
    serve_parser.add_argument("--workspace", default=None, help="Optional workspace root directory to anchor relative file paths")

    # test command
    test_parser = subparsers.add_parser("test", help="Test pure HTTP connection to MiMo")
    test_parser.add_argument("--prompt", default="Halo, siapa kamu?", help="Prompt to test")
    test_parser.add_argument("--session-file", default=None, help="Custom path to session.json")

    # login command
    login_parser = subparsers.add_parser("login", help="Interactive one-time browser login")
    login_parser.add_argument("--session-file", default=None, help="Custom destination path for session.json")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "test":
        cmd_test(args)
    elif args.command == "login":
        cmd_login(args)


if __name__ == "__main__":
    main()
