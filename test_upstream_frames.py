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
        if "ConnectionRefused" in repr(e) or "10061" in repr(e):
            print("  SKIP  server not running (expected in CI/verify)")
        else:
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

    # --- Unit: URL masking must defuse Xiaomi's server-side webSearch injector ---
    print("[unit] URL masking for upstream (webSearch injection guard)")
    mask = getattr(m, "mask_urls_for_upstream", None)
    check("mask_urls_for_upstream exists", mask is not None,
          "(missing -> URL masking not implemented)")
    if mask is not None:
        raw = "Buka https://en.wikipedia.org/wiki/Xiaomi lalu ringkas."
        masked = mask(raw)
        # The literal URL must no longer be present verbatim ...
        check("literal URL removed", "https://en.wikipedia.org/wiki/Xiaomi" not in masked,
              f"got {masked!r}")
        # ... but the pieces must still be readable and reassemble to the original.
        check("reassembles to original URL",
              masked.replace(" . ", ".").replace(" / ", "/")
              .split("Buka ")[1].split(" lalu")[0] == "https://en.wikipedia.org/wiki/Xiaomi",
              f"got {masked!r}")
        check("scheme kept intact", masked.startswith("Buka https://"), f"got {masked!r}")
        # Text without a URL is untouched.
        check("plain text untouched", mask("Halo, apa kabar?") == "Halo, apa kabar?")
        # Tool observations (already-fetched URLs the model may re-use) untouched.
        check("non-http text untouched", mask("lihat example.com saja") == "lihat example.com saja")

    # --- Unit: masking applied inside build_agent_prompt user turns ---
    print("[unit] build_agent_prompt masks user URLs")
    if mask is not None:
        msgs = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Buka https://example.com lalu ringkas."},
        ]
        prompt = m.build_agent_prompt(msgs, tools=[{
            "type": "function",
            "function": {"name": "browser_navigate", "description": "Navigate",
                         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}},
        }], framework="default")
        check("user URL masked in built prompt", "https://example.com" not in prompt,
              f"found literal URL in prompt")
        check("masked URL still readable", "example . com" in prompt)

    # --- Unit: coerce_tool_args repairs JSON-stringified array/object params ---
    # Bug: MiMo sometimes emits a tool argument that SHOULD be an array as a JSON
    # *string* (e.g. queries='["a","b"]' instead of ["a","b"]). Hermes' tool_search
    # / tool_describe then reject it ("requires a 'name'" / not_found), the model
    # gives up on tools, and no MCP tool is ever called.
    print("[unit] coerce_tool_args repairs stringified array/object params")
    coerce = getattr(m, "coerce_tool_args", None)
    check("coerce_tool_args exists", coerce is not None)
    if coerce is not None:
        tools_schema = [
            {"type": "function", "function": {"name": "tool_search",
             "parameters": {"type": "object", "properties": {
                 "queries": {"type": "array", "items": {"type": "string"}},
                 "limit": {"type": "integer"}}, "required": ["queries"]}}},
            {"type": "function", "function": {"name": "tool_describe",
             "parameters": {"type": "object", "properties": {
                 "names": {"type": "array", "items": {"type": "string"}}}}}},
        ]
        got = coerce("tool_search",
                     {"queries": '["open browser", "take screenshot"]', "limit": 10},
                     tools_schema)
        check("stringified array -> list", isinstance(got.get("queries"), list)
              and got["queries"] == ["open browser", "take screenshot"], f"got {got!r}")
        check("non-string param untouched", got.get("limit") == 10, f"got {got!r}")

        got2 = coerce("tool_describe", {"names": '["mcp__playwright__browser_navigate"]'},
                      tools_schema)
        check("describe names string -> list",
              got2.get("names") == ["mcp__playwright__browser_navigate"], f"got {got2!r}")

        # A genuine list must survive untouched.
        got3 = coerce("tool_search", {"queries": ["a", "b"]}, tools_schema)
        check("real list untouched", got3.get("queries") == ["a", "b"], f"got {got3!r}")

        # Bare (non-JSON) string for an array param is wrapped, not dropped.
        got4 = coerce("tool_search", {"queries": "open browser"}, tools_schema)
        check("bare string wrapped into list", got4.get("queries") == ["open browser"],
              f"got {got4!r}")

        # Unknown tool / no schema: leave the dict alone.
        got5 = coerce("mystery", {"x": "[1,2]"}, tools_schema)
        check("unknown tool untouched", got5 == {"x": "[1,2]"}, f"got {got5!r}")

        # Array-of-objects alias: tool_call wants calls=[{name, arguments}];
        # MiMo emits calls=[{tool, arguments}] -> rename tool -> name.
        schema_call = [{"type": "function", "function": {"name": "tool_call",
            "parameters": {"type": "object", "properties": {
                "calls": {"type": "array", "items": {"type": "object", "properties": {
                    "name": {"type": "string"}, "arguments": {"type": "object"}}}}}}}}]
        got6 = coerce("tool_call",
                      {"calls": [{"tool": "mcp__playwright__browser_navigate",
                                  "arguments": {"url": "https://x"}}]}, schema_call)
        check("calls[].tool renamed to name",
              got6["calls"][0].get("name") == "mcp__playwright__browser_navigate"
              and "tool" not in got6["calls"][0], f"got {got6!r}")

    # --- Integration: XML tool_call with stringified array is repaired ---
    print("[integration] smart_extract_tool_calls repairs stringified array param")
    if coerce is not None:
        reply = (
            "<tool_call>\n"
            "<function=tool_search>\n"
            '<parameter=queries>["open browser", "screenshot"]</parameter>\n'
            "<parameter=limit>10</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        tools_schema = [
            {"type": "function", "function": {"name": "tool_search",
             "parameters": {"type": "object", "properties": {
                 "queries": {"type": "array", "items": {"type": "string"}},
                 "limit": {"type": "integer"}}}}},
        ]
        calls, _ = m.smart_extract_tool_calls(reply, "buka browser", tools_schema)
        ok = (len(calls) == 1 and calls[0]["name"] == "tool_search"
              and isinstance(calls[0]["arguments"].get("queries"), list))
        check("extracted queries is a list", ok, f"got {calls!r}")

    # ------------------------------------------------------------------
    # [unit] _STATS global + /v1/stats endpoint
    # ------------------------------------------------------------------
    print("[unit] stats tracking")

    # Reset stats for deterministic test
    m._STATS["requests"] = 0
    m._STATS["errors"] = 0
    m._STATS["tool_calls"] = 0
    m._STATS["coerced_args"] = 0
    m._STATS["tool_usage"] = {}
    m._STATS["model_usage"] = {}

    check("_STATS exists", hasattr(m, "_STATS"), "missing global")
    check("_START_TIME exists", hasattr(m, "_START_TIME"), "missing global")
    check("_STATS keys complete",
          all(k in m._STATS for k in ("requests", "errors", "tool_calls",
                                       "coerced_args", "tool_usage", "model_usage")),
          f"got {list(m._STATS)}")

    # coerce_tool_args should increment coerced_args when it repairs
    bad_args = {"queries": '["a","b"]'}
    schema_tools = [{"type": "function", "function": {"name": "tool_search",
        "parameters": {"type": "object", "properties": {
            "queries": {"type": "array", "items": {"type": "string"}}}}}}]
    m.coerce_tool_args("tool_search", bad_args, schema_tools)
    check("coerced_args incremented", m._STATS["coerced_args"] == 1,
          f"got {m._STATS['coerced_args']}")

    # no-op coercion should NOT increment
    good_args = {"queries": ["a", "b"]}
    m.coerce_tool_args("tool_search", good_args, schema_tools)
    check("coerced_args unchanged for valid args", m._STATS["coerced_args"] == 1,
          f"got {m._STATS['coerced_args']}")

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
