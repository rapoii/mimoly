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
    from fastapi.responses import JSONResponse, StreamingResponse
    import httpx
    import uvicorn
except ImportError:
    print("[mimoly] Missing dependencies. Run: pip install -r requirements.txt")
    sys.exit(1)

BASE_DIR = Path(__file__).parent.resolve()
SESSION_FILE = BASE_DIR / "session.json"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
CHAT_API_URL = "https://aistudio.xiaomimimo.com/open-apis/bot/chat"
CHAT_CONV_SAVE_URL = "https://aistudio.xiaomimimo.com/open-apis/chat/conversation/save"

app = FastAPI(title="Mimoly Web2API Proxy", description="100% Pure HTTP OpenAI-Compatible Proxy for Xiaomi MiMo")

# Allow CORS for web frontends (Cherry Studio, NextChat, OpenRouter, 9router, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def build_agent_prompt(messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> str:
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
            snippet = content[:2500] + ("..." if len(content) > 2500 else "")
            history.append(f"[Hasil {tool_name}]: {snippet}")

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

        prompt_lines.append(
            "Kamu adalah asisten AI dengan kemampuan tool calling.\n"
            "Berikut tools yang tersedia:\n" + "\n".join(tool_desc) + "\n\n"
            "Jika perlu memanggil tool, panggil dengan format JSON:\n"
            "```json\n"
            '{"tool_calls": [{"name": "nama_tool", "arguments": {"param1": "nilai"}}]}\n'
            "```\n"
            "atau:\n"
            "<tool_call>\n"
            '{"name": "nama_tool", "arguments": {"param1": "nilai"}}\n'
            "</tool_call>\n\n"
            "Jika menulis kode program, tuliskan kode lengkap fungsional sekarang juga."
        )

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


def smart_extract_tool_calls(reply_text: str, user_prompt: str, available_tools: Optional[List[Dict[str, Any]]] = None):
    """Detect explicit JSON tool_calls or extract code blocks mapped to write/shell tools."""
    tool_names = [t.get("function", {}).get("name") for t in available_tools] if available_tools else []

    # 1. Cek explicit JSON tool_calls
    # Pola 1A: <tool_call> ... </tool_call>
    m_tc = re.findall(r"<tool_call>([\s\S]*?)(?:</tool_call>|$)", reply_text)
    if m_tc:
        tc_list = []
        for block in m_tc:
            clean_block = block.replace("</think>", "").strip()
            # Match json object inside
            m_json_obj = re.search(r"(\{[\s\S]*\})", clean_block)
            if m_json_obj:
                try:
                    tc_obj = json.loads(m_json_obj.group(1), strict=False)
                    if "tool_calls" in tc_obj and isinstance(tc_obj["tool_calls"], list):
                        for item in tc_obj["tool_calls"]:
                            name = item.get("name")
                            args = item.get("arguments") or item.get("parameters") or {}
                            if name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                                name = "terminal"
                            tc_list.append({"name": name, "arguments": args})
                    elif "name" in tc_obj:
                        name = tc_obj["name"]
                        args = tc_obj.get("arguments") or tc_obj.get("parameters") or {}
                        if name in ["RunCommand", "run_command", "bash", "shell"] and "terminal" in tool_names:
                            name = "terminal"
                        tc_list.append({"name": name, "arguments": args})
                except Exception:
                    pass
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
                return d["tool_calls"], clean
            elif "name" in d and ("parameters" in d or "arguments" in d):
                args = d.get("parameters") or d.get("arguments") or {}
                clean = reply_text.replace(m_json.group(0), "").strip()
                return [{"name": d["name"], "arguments": args}], clean
        except Exception:
            pass

    m2 = re.search(r"\{\s*\"tool_calls\"\s*:\s*(\[[\s\S]*?\])\s*\}", reply_text)
    if m2:
        try:
            arr = json.loads(m2.group(1), strict=False)
            calls = []
            for item in arr:
                if "name" in item:
                    calls.append({"name": item.get("name"), "arguments": item.get("arguments") or item.get("parameters") or {}})
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


@app.get("/v1/models")
@app.get("/models")
@app.get("/api/v1/models")
async def list_models():
    now = int(time.time())
    model_list = [
        {"id": "mimo-v2.5-pro", "object": "model", "created": now, "owned_by": "xiaomi"},
        {"id": "custom/mimo-v2.5-pro", "object": "model", "created": now, "owned_by": "xiaomi"},
        {"id": "mimo-v2.5", "object": "model", "created": now, "owned_by": "xiaomi"},
        {"id": "mimo", "object": "model", "created": now, "owned_by": "xiaomi"},
        {"id": "claude-fable-5", "object": "model", "created": now, "owned_by": "xiaomi"},
    ]
    return {"object": "list", "data": model_list}


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str):
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "xiaomi"
    }


@app.get("/api/tags")
async def ollama_tags():
    return {
        "models": [
            {
                "name": "mimo-v2.5-pro",
                "model": "mimo-v2.5-pro",
                "modified_at": "2026-09-28T00:00:00Z",
                "size": 7000000000,
                "digest": "sha256:mimoly",
                "details": {"format": "gguf", "family": "mimo"}
            }
        ]
    }


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
    model = body.get("model", "mimo-v2.5-pro")

    print(f"[mimoly] Incoming request: model={model}, stream={stream}, messages={len(messages)}, tools={len(tools) if tools else 0}")
    try:
        with open("last_request.json", "w", encoding="utf-8") as f_req:
            json.dump(body, f_req, indent=2, ensure_ascii=False)
    except Exception:
        pass

    # Load authentic session cookies
    try:
        cookies = get_session_cookies()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=401)

    # Format unified prompt
    if tools or len(messages) > 1:
        user_prompt = build_agent_prompt(messages, tools)
    else:
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                c = m.get("content", "")
                user_prompt = c if isinstance(c, str) else str(c)
                break

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
    try:
        async with httpx.AsyncClient(timeout=10.0) as save_client:
            await save_client.post(
                save_url,
                headers=headers,
                cookies=cookies,
                json={"conversationId": conv_id, "type": "chat", "title": "Mimoly Session"}
            )
    except Exception as e:
        print(f"[mimoly] Warning: failed to save conversation: {e}")

    # Xiaomi payload structure
    upstream_payload = {
        "msgId": uuid.uuid4().hex,
        "conversationId": conv_id,
        "query": user_prompt,
        "isEditedQuery": False,
        "modelConfig": {
            "enableThinking": True,
            "webSearchStatus": "disabled",
            "model": "mimo-v2.5-pro",
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
                                continue

                            content_piece = parsed.get("content", "")
                            # Ignore dialog ID event
                            if content_piece and content_piece.isdigit() and len(content_piece) >= 7:
                                continue

                            if not content_piece:
                                continue

                            clean_piece = content_piece.replace("<think>\x00", "").replace("<think>", "").replace("</think>", "").replace("\x00", "")
                            if not clean_piece:
                                continue

                            # If tools are requested, we buffer to check for tool calls
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
                                        "delta": {"content": clean_piece},
                                        "finish_reason": None
                                    }]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                                accumulated_chunks.append(clean_piece)

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
                        continue

                    content_piece = parsed.get("content", "")
                    if content_piece and content_piece.isdigit() and len(content_piece) >= 7:
                        continue
                    if content_piece:
                        accumulated_chunks.append(content_piece.replace("\x00", ""))

    except Exception as e:
        return JSONResponse({"error": f"Failed to connect to upstream: {e}"}, status_code=502)

    full_reply = "".join(accumulated_chunks)

    # Detect tool calls
    raw_calls, clean_text = smart_extract_tool_calls(full_reply, user_prompt, tools)
    openai_tool_calls = []
    finish_reason = "stop"

    if raw_calls:
        finish_reason = "tool_calls"
        for i, tc in enumerate(raw_calls):
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
