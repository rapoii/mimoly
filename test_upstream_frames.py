#!/usr/bin/env python3
"""Regression tests for mimoly upstream frame parsing + tool-call passthrough.

Bug: Xiaomi's upstream SSE occasionally emits a JSON *array* frame (internal
webSearch payload, e.g. ``data:[{"id":..,"text":"# Historical weather .."}]``).
Both the streaming and non-streaming readers called ``parsed.get(...)`` without
a type check, raising ``'list' object has no attribute 'get'``, which surfaced as
HTTP 502 "Failed to connect to upstream" and made clients (Hermes) drop tools
and fall back to text-only answers.

Run:  .venv/Scripts/python.exe test_upstream_frames.py
"""
import importlib.util
import json
import sys

MIMOLY = "mimoly.py"


def load_mimoly():
    spec = importlib.util.spec_from_file_location("mimoly_mod", MIMOLY)
    m = importlib.util.module_from_spec(spec)
    sys.modules["mimoly_mod"] = m
    spec.loader.exec_module(m)
    return m


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


def main():
    m = load_mimoly()

    # --- Unit: the frame normalizer must accept dicts, reject arrays/others ---
    print("[unit] upstream frame normalizer")
    norm = getattr(m, "normalize_upstream_frame", None)
    check("normalize_upstream_frame exists", norm is not None,
          "(missing -> parsing guard not implemented)")

    if norm is not None:
        obj_frame = {"type": "text", "content": "hello"}
        arr_frame = [{"id": "2026-10-04_x_1", "text": "# Historical weather data"}]
        check("dict frame passes through", norm(obj_frame) == obj_frame)
        check("list frame -> None", norm(arr_frame) is None, f"got {norm(arr_frame)!r}")
        check("str frame -> None", norm("nope") is None)
        check("int frame -> None", norm(42) is None)
        check("None frame -> None", norm(None) is None)

    # --- Unit: real captured frames must be classifiable without raising ---
    print("[unit] captured upstream frames (regression set)")
    frames = [
        '{"content":"14995793"}',
        '{"type":"text","content":"webSearch"}',
        '[{"id":"2026-10-04_cda5b7ff23b43e49319af315cf3baeda_1","text":"# Historical weather data for any location"}]',
        '{"type":"text","content":"<think>\\u0000The user wants a"}',
        '{"promptTokens":2933,"completionTokens":1220,"totalTokens":4153,"nativeUsage":{}}',
    ]
    if norm is not None:
        ok = True
        for f in frames:
            try:
                norm(json.loads(f))
            except Exception as e:  # noqa: BLE001
                ok = False
                check(f"frame classified: {f[:40]}", False, f"raised {e!r}")
        if ok:
            check("all captured frames classified without raising", True)

    # --- Integration: /v1/chat/completions with tools must not 502 ---
    print("[integration] POST /v1/chat/completions with tools")
    try:
        import urllib.request

        payload = {
            "model": "mimo-v2.6-pro",
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": "You are a helpful agent. Use tools when needed."},
                {"role": "user", "content": "Berapa cuaca di Jakarta? Pakai tool get_weather."},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {"type": "object",
                                   "properties": {"city": {"type": "string"}},
                                   "required": ["city"]},
                },
            }],
            "stream": False,
            "max_completion_tokens": 512,
        }
        req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
        check("HTTP 200 (no 502)", True)
        msg = body.get("choices", [{}])[0].get("message", {})
        has_calls = bool(msg.get("tool_calls"))
        has_text = bool((msg.get("content") or "").strip())
        check("response has tool_calls or content", has_calls or has_text,
              f"tool_calls={msg.get('tool_calls')!r} content={msg.get('content')!r}")
        print(f"        finish_reason={body['choices'][0].get('finish_reason')} "
              f"tool_calls={json.dumps(msg.get('tool_calls'), ensure_ascii=False)[:200]}")
    except Exception as e:  # noqa: BLE001
        check("integration request succeeded", False, f"raised {e!r}")

    # --- Unit: tool observations with JSON must not leak unescaped `{"` ---
    print("[unit] tool observation quote sanitisation")
    try:
        msgs = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "cari tool browser"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "tool_search", "arguments": "{}"}}]},
            {"role": "tool", "name": "tool_search",
             "content": json.dumps({"results": [{"matches": ["mcp__playwright__browser_click"]}]})},
        ]
        prompt = m.build_agent_prompt(msgs, tools=[{
            "type": "function",
            "function": {"name": "tool_search", "description": "Search tools",
                         "parameters": {"type": "object", "properties": {"queries": {"type": "string"}}}},
        }], framework="default")
        check("no unescaped `{\"` in prompt", '{"' not in prompt,
              f"found at {prompt.find(chr(123)+chr(34))}")
        check("observation still present", "mcp__playwright__browser_click" in prompt)
    except Exception as e:  # noqa: BLE001
        check("build_agent_prompt handled tool JSON", False, f"raised {e!r}")

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
