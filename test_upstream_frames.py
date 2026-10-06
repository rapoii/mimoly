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
import os
import sys
import urllib.request

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
    if not (os.environ.get("RUN_INTEGRATION") or "--integration" in sys.argv):
        print("  SKIP  live integration test (opt-in via --integration or RUN_INTEGRATION=1)")
    else:
        import socket
        server_online = False
        try:
            sock = socket.create_connection(("127.0.0.1", 8080), timeout=0.3)
            sock.close()
            server_online = True
        except Exception:
            server_online = False

        if not server_online:
            print("  SKIP  server not running")
        else:
            try:
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
                with urllib.request.urlopen(req, timeout=60) as resp:
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

    # ------------------------------------------------------------------
    # [unit] coerce_tool_args — proactive patterns + schema validation
    # ------------------------------------------------------------------
    print("[unit] coerce_tool_args proactive")

    def _schema(props, required=None):
        return [{"type": "function", "function": {"name": "t",
                 "parameters": {"type": "object", "properties": props,
                                "required": required or []}}}]

    # number: "3.14" -> 3.14
    a = {"ratio": "3.14"}
    m.coerce_tool_args("t", a, _schema({"ratio": {"type": "number"}}))
    check("number coerced from string",
          a["ratio"] == 3.14 and isinstance(a["ratio"], float), f"got {a['ratio']!r}")

    # boolean: "1"/"0"/int 1 -> True/False/True
    a = {"flag": "1"}
    m.coerce_tool_args("t", a, _schema({"flag": {"type": "boolean"}}))
    check("boolean from '1'", a["flag"] is True, f"got {a['flag']!r}")
    a = {"flag": "0"}
    m.coerce_tool_args("t", a, _schema({"flag": {"type": "boolean"}}))
    check("boolean from '0'", a["flag"] is False, f"got {a['flag']!r}")
    a = {"flag": 1}
    m.coerce_tool_args("t", a, _schema({"flag": {"type": "boolean"}}))
    check("boolean from int 1", a["flag"] is True, f"got {a['flag']!r}")

    # nested object recursion: config.level "3" -> 3
    a = {"config": '{"level": "3"}'}
    m.coerce_tool_args("t", a, _schema({"config": {"type": "object", "properties": {
        "level": {"type": "integer"}}}}))
    check("nested object scalar coerced",
          a["config"] == {"level": 3}, f"got {a['config']!r}")

    # null string -> None for nullable field
    a = {"note": "null"}
    m.coerce_tool_args("t", a, _schema({"note": {"type": ["string", "null"]}}))
    check("'null' string -> None", a["note"] is None, f"got {a['note']!r}")

    # --- schema validation ---
    check("validate_tool_args exists", hasattr(m, "validate_tool_args"), "missing")
    v = m.validate_tool_args("t", {"ratio": "x"},
                             _schema({"ratio": {"type": "number"}}, ["ratio"]))
    check("validate returns dict", isinstance(v, dict), f"got {type(v)}")
    check("validate flags type mismatch",
          v.get("valid") is False and bool(v.get("issues")), f"got {v}")
    v2 = m.validate_tool_args("t", {},
                              _schema({"ratio": {"type": "number"}}, ["ratio"]))
    check("validate flags missing required",
          v2.get("valid") is False
          and any("required" in i.lower() or "missing" in i.lower()
                  for i in v2.get("issues", [])), f"got {v2}")
    v3 = m.validate_tool_args("t", {"ratio": 1.5},
                              _schema({"ratio": {"type": "number"}}, ["ratio"]))
    check("validate passes good args", v3.get("valid") is True, f"got {v3}")

    # invalid_args counter increments when normalize_tool_args sees a mismatch
    m._STATS["invalid_args"] = 0
    m.normalize_tool_args("t", {"ratio": "not-a-number"},
                          available_tools=_schema({"ratio": {"type": "number"}}, ["ratio"]))
    check("invalid_args counted after failed coercion",
          m._STATS["invalid_args"] == 1, f"got {m._STATS['invalid_args']}")

    # a clean, coercible arg should NOT count as invalid
    m._STATS["invalid_args"] = 0
    m.normalize_tool_args("t", {"ratio": "2.5"},
                          available_tools=_schema({"ratio": {"type": "number"}}, ["ratio"]))
    check("coercible arg not counted invalid",
          m._STATS["invalid_args"] == 0, f"got {m._STATS['invalid_args']}")

    # ------------------------------------------------------------------
    # [unit] token usage tracking (_record_usage)
    # ------------------------------------------------------------------
    print("[unit] token usage tracking")

    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        m._STATS[k] = 0
    m._STATS["model_tokens"] = {}

    check("token keys present",
          all(k in m._STATS for k in ("prompt_tokens", "completion_tokens",
                                       "total_tokens", "reasoning_tokens", "model_tokens")),
          f"got {list(m._STATS)}")

    m._record_usage({"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                    "model-a")
    check("prompt_tokens accumulated", m._STATS["prompt_tokens"] == 100,
          f"got {m._STATS['prompt_tokens']}")
    check("completion_tokens accumulated", m._STATS["completion_tokens"] == 20,
          f"got {m._STATS['completion_tokens']}")
    check("total_tokens accumulated", m._STATS["total_tokens"] == 120,
          f"got {m._STATS['total_tokens']}")
    check("per-model tokens recorded",
          m._STATS["model_tokens"].get("model-a", {}).get("total_tokens") == 120,
          f"got {m._STATS['model_tokens']}")

    # reasoning tokens extracted from nested detail
    m._record_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                     "completion_tokens_details": {"reasoning_tokens": 3}}, "model-a")
    check("reasoning_tokens accumulated", m._STATS["reasoning_tokens"] == 3,
          f"got {m._STATS['reasoning_tokens']}")
    check("per-model accumulates across calls",
          m._STATS["model_tokens"]["model-a"]["total_tokens"] == 135,
          f"got {m._STATS['model_tokens']['model-a']}")

    # missing keys default to 0, no crash
    m._record_usage({}, "model-b")
    check("empty usage is safe", m._STATS["model_tokens"]["model-b"]["total_tokens"] == 0,
          f"got {m._STATS['model_tokens'].get('model-b')}")

    # ------------------------------------------------------------------
    # [unit] /dashboard endpoint serves HTML
    # ------------------------------------------------------------------
    print("[unit] dashboard endpoint")
    check("_DASHBOARD_HTML defined", hasattr(m, "_DASHBOARD_HTML"), "missing")
    check("dashboard HTML non-trivial",
          len(getattr(m, "_DASHBOARD_HTML", "")) > 1000,
          f"len={len(getattr(m, '_DASHBOARD_HTML', ''))}")
    check("dashboard fetches /v1/stats", "/v1/stats" in getattr(m, "_DASHBOARD_HTML", ""),
          "no stats fetch")
    check("dashboard route registered",
          any(getattr(r, "path", "") == "/dashboard" for r in m.app.routes),
          "route not found")

    # ------------------------------------------------------------------
    # [unit] analytics: latency, finish_reasons, timeline
    # ------------------------------------------------------------------
    print("[unit] analytics tracking")

    check("analytics keys present",
          all(k in m._STATS for k in ("finish_reasons", "latency_ms", "ttft_ms", "timeline")),
          f"got {list(m._STATS)}")
    check("latency helper exists", hasattr(m, "_record_latency"), "missing")
    check("finish_reason helper exists", hasattr(m, "_record_finish_reason"), "missing")
    check("timeline helper exists", hasattr(m, "_record_timeline"), "missing")

    # _record_latency accumulates min/max/avg correctly
    m._STATS["latency_ms"] = {"count": 0, "sum": 0.0, "min": None, "max": None}
    m._record_latency("latency_ms", 100.0)
    m._record_latency("latency_ms", 300.0)
    m._record_latency("latency_ms", 200.0)
    lb = m._STATS["latency_ms"]
    check("latency count", lb["count"] == 3, f"got {lb}")
    check("latency sum", lb["sum"] == 600.0, f"got {lb}")
    check("latency min", lb["min"] == 100.0, f"got {lb}")
    check("latency max", lb["max"] == 300.0, f"got {lb}")

    # _record_finish_reason counts per reason
    m._STATS["finish_reasons"] = {}
    m._record_finish_reason("stop")
    m._record_finish_reason("stop")
    m._record_finish_reason("tool_calls")
    m._record_finish_reason("")   # empty ignored
    check("finish_reasons counted",
          m._STATS["finish_reasons"] == {"stop": 2, "tool_calls": 1},
          f"got {m._STATS['finish_reasons']}")

    # _record_timeline buckets by minute + accumulates
    m._STATS["timeline"] = {}
    m._record_timeline(requests=1, total_tokens=10)
    m._record_timeline(requests=2, tool_calls=1, total_tokens=5)
    m._record_timeline(errors=1)
    tl = m._STATS["timeline"]
    check("timeline single minute bucket", len(tl) == 1, f"got {len(tl)} buckets")
    bucket = list(tl.values())[0]
    check("timeline requests accumulated", bucket["requests"] == 3, f"got {bucket}")
    check("timeline tokens accumulated", bucket["total_tokens"] == 15, f"got {bucket}")
    check("timeline tool_calls accumulated", bucket["tool_calls"] == 1, f"got {bucket}")
    check("timeline errors accumulated", bucket["errors"] == 1, f"got {bucket}")

    # _record_usage also feeds the timeline
    m._STATS["timeline"] = {}
    m._record_usage({"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}, "m")
    check("usage feeds timeline tokens",
          list(m._STATS["timeline"].values())[0]["total_tokens"] == 10,
          f"got {m._STATS['timeline']}")

    # /v1/stats endpoint exposes the new analytics fields
    import asyncio as _asyncio
    _stats_resp = _asyncio.run(m.stats())
    check("stats exposes finish_reasons", "finish_reasons" in _stats_resp, f"keys={list(_stats_resp)}")
    check("stats exposes latency summary",
          "latency" in _stats_resp and "avg_ms" in _stats_resp["latency"], f"got {_stats_resp.get('latency')}")
    check("stats exposes ttft summary", "ttft" in _stats_resp, f"keys={list(_stats_resp)}")
    check("stats exposes timeline list",
          isinstance(_stats_resp.get("timeline"), list), f"got {type(_stats_resp.get('timeline'))}")

    # ------------------------------------------------------------------
    # [unit] dashboard: neobrutalism + charts
    # ------------------------------------------------------------------
    print("[unit] dashboard neobrutalism + charts")
    _dash = getattr(m, "_DASHBOARD_HTML", "")
    check("dashboard loads Chart.js", "chart.js" in _dash.lower(), "no chart lib")
    check("dashboard has timeline canvas", "c-timeline" in _dash, "missing timeline chart")
    check("dashboard has finish-reasons chart", "c-finish" in _dash, "missing finish chart")
    check("dashboard has token-split chart", "c-tokens" in _dash, "missing token chart")
    check("dashboard uses neobrutalism hard shadow", "6px 6px 0" in _dash, "no hard shadow")
    check("dashboard has thick ink borders", "3px solid" in _dash, "no thick border")
    check("dashboard offline fallback", "offline-note" in _dash, "no fallback")

    # ------------------------------------------------------------------
    # [unit] per-request log (input + output per request)
    # ------------------------------------------------------------------
    print("[unit] per-request request log")
    check("_REQUEST_LOG exists", hasattr(m, "_REQUEST_LOG"), "missing ring buffer")
    check("_new_request helper exists", hasattr(m, "_new_request"), "missing")
    check("_finish_request helper exists", hasattr(m, "_finish_request"), "missing")
    check("_clip helper exists", hasattr(m, "_clip"), "missing")
    check("_last_user_text helper exists", hasattr(m, "_last_user_text"), "missing")

    if hasattr(m, "_clip"):
        check("_clip truncates long text", len(m._clip("x" * 5000, 100)) <= 120,
              f"got {len(m._clip('x'*5000, 100))}")
        check("_clip keeps short text", m._clip("hello", 100) == "hello")

    if hasattr(m, "_last_user_text"):
        check("_last_user_text picks last user msg",
              m._last_user_text([{"role": "system", "content": "s"},
                                 {"role": "user", "content": "a"},
                                 {"role": "user", "content": "b"}]) == "b",
              "wrong pick")
        check("_last_user_text handles multimodal",
              m._last_user_text([{"role": "user", "content": [
                  {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {}}]}]) == "hi",
              "multimodal not handled")
        check("_last_user_text empty when none", m._last_user_text([{"role": "system", "content": "s"}]) == "")

    if hasattr(m, "_REQUEST_LOG") and hasattr(m, "_new_request"):
        m._REQUEST_LOG.clear()
        rec = m._new_request(model="mimo", stream=True, messages=2, tools=1,
                             prompt="halo", tools_names=["get_weather"])
        check("record has id + ts", bool(rec.get("id")) and bool(rec.get("ts")), f"got {rec}")
        check("record stored in ring buffer", len(m._REQUEST_LOG) == 1, f"got {len(m._REQUEST_LOG)}")
        check("record keeps input", rec.get("input") == "halo", f"got {rec.get('input')}")
        check("record defaults output empty", rec.get("output") == "", f"got {rec.get('output')}")

        # _finish_request fills the outcome fields
        if hasattr(m, "_finish_request"):
            m._finish_request(rec, output="jawaban", finish_reason="stop", latency_ms=123.4,
                              ttft_ms=45.6, usage={"prompt_tokens": 10, "completion_tokens": 5,
                                                   "total_tokens": 15}, tools_called=[])
            check("finish sets output", rec.get("output") == "jawaban", f"got {rec.get('output')}")
            check("finish sets finish_reason", rec.get("finish_reason") == "stop")
            check("finish sets latency", rec.get("latency_ms") == 123.4, f"got {rec.get('latency_ms')}")
            check("finish sets tokens", rec.get("prompt_tokens") == 10 and rec.get("completion_tokens") == 5,
                  f"got {rec}")
            check("finish sets tools_called", rec.get("tools_called") == [])

        # ring buffer must be bounded (deque maxlen)
        m._REQUEST_LOG.clear()
        for i in range(120):
            m._new_request(model="m", stream=False, messages=1, tools=0,
                           prompt=str(i), tools_names=[])
        check("ring buffer bounded to <=50", len(m._REQUEST_LOG) <= 50,
              f"grew to {len(m._REQUEST_LOG)}")

    # /v1/stats must expose the recent request log (newest first)
    _sr = _asyncio.run(m.stats())
    check("stats exposes recent_requests list",
          isinstance(_sr.get("recent_requests"), list), f"got {type(_sr.get('recent_requests'))}")

    check("dashboard has recent-requests panel",
          "recent" in _dash.lower(), "no request log panel")

    # ------------------------------------------------------------------
    # [unit] streaming safety-net (cancelled streams must still finalize)
    # ------------------------------------------------------------------
    print("[unit] streaming cancel safety-net")
    _src = open(MIMOLY, encoding="utf-8").read()
    check("stream generator has finally safety-net", "finally:" in _src, "no finally")
    check("safety-net records 'cancelled'", '"cancelled"' in _src, "no cancelled reason")
    check("safety-net guarded by finish_reason check",
          'not _req_rec.get("finish_reason")' in _src, "no guard -> double counting")
    # _finish_request must tolerate None (defensive)
    m._finish_request(None, output="x")  # must not raise
    check("_finish_request tolerates None rec", True)
    # _clip is bounded even for huge outputs
    check("_clip bounds huge output", len(m._clip("y" * 100000, 600)) <= 605,
          f"got {len(m._clip('y'*100000, 600))}")

    # ------------------------------------------------------------------
    # [unit] dynamic and portable path normalization
    # ------------------------------------------------------------------
    print("[unit] dynamic & portable tool path normalization")
    # 1) When MIMOLY_WORKSPACE is unset: relative path is preserved as-is (clean slashes)
    os.environ.pop("MIMOLY_WORKSPACE", None)
    res_raw = m.normalize_tool_args("write_file", {"path": "src\\utils\\helper.ts"})
    check("relative path preserved without workspace injection", res_raw.get("path") == "src/utils/helper.ts",
          f"got {res_raw.get('path')}")

    # 2) When MIMOLY_WORKSPACE is set: relative path anchored to workspace
    os.environ["MIMOLY_WORKSPACE"] = "/custom/project/root"
    res_anchored = m.normalize_tool_args("write_file", {"path": "src\\index.ts"})
    check("relative path anchored to custom workspace", res_anchored.get("path") == "/custom/project/root/src/index.ts",
          f"got {res_anchored.get('path')}")

    # 3) Absolute paths never double-prefixed
    res_abs = m.normalize_tool_args("read_file", {"path": "/custom/project/root/config.json"})
    check("absolute workspace path not double-prefixed", res_abs.get("path") == "/custom/project/root/config.json",
          f"got {res_abs.get('path')}")
    os.environ.pop("MIMOLY_WORKSPACE", None)

    # 4) Zero hardcoded machine/user paths in mimoly.py
    for forbidden in ["Hermes Workspace", "D:/Software", "D:\\Software", "spectra"]:
        check(f"zero hardcoded '{forbidden}' in source", forbidden not in _src,
              f"found '{forbidden}' in {MIMOLY}")

    # ------------------------------------------------------------------
    # [unit] auto-healing and schema drift recovery (A11, A12, A38, A39)
    # ------------------------------------------------------------------
    print("[unit] tool argument auto-healing & schema drift recovery")
    mock_tools = [
        {"type": "function", "function": {
            "name": "execute_code",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
        }},
        {"type": "function", "function": {
            "name": "write_file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}
        }},
        {"type": "function", "function": {
            "name": "tool_call",
            "parameters": {"type": "object", "properties": {"calls": {"type": "array", "items": {"type": "object"}}}, "required": ["calls"]}
        }},
        {"type": "function", "function": {
            "name": "browser_evaluate",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        }}
    ]

    # Test 1: execute_code heals 'command' -> 'code'
    healed_exec = m.coerce_tool_args("execute_code", {"command": "print(123)"}, mock_tools)
    check("execute_code alias healed command->code", healed_exec.get("code") == "print(123)", f"got {healed_exec}")

    # Test 2: write_file heals 'filepath' -> 'path' and 'text' -> 'content'
    healed_write = m.coerce_tool_args("write_file", {"filepath": "foo.txt", "text": "hello"}, mock_tools)
    check("write_file alias healed filepath & text", healed_write.get("path") == "foo.txt" and healed_write.get("content") == "hello", f"got {healed_write}")

    # Test 3: write_file recovers top-level JSON fields when content is missing (A11)
    healed_pkg = m.normalize_tool_args("write_file", {"path": "package.json", "name": "my-pkg", "version": "1.0"}, "", mock_tools)
    check("write_file recovers top-level json fields into content", "my-pkg" in healed_pkg.get("content", ""), f"got {healed_pkg}")

    # Test 4: tool_call wraps dict into list for array parameter
    healed_call = m.coerce_tool_args("tool_call", {"calls": {"name": "test", "arguments": {}}}, mock_tools)
    check("tool_call wraps dict into list for calls", isinstance(healed_call.get("calls"), list), f"got {healed_call}")

    # Test 5: browser_evaluate heals 'script' -> 'expression'
    healed_eval = m.coerce_tool_args("browser_evaluate", {"script": "1+1"}, mock_tools)
    check("browser_evaluate heals script->expression", healed_eval.get("expression") == "1+1", f"got {healed_eval}")

    # Test 6: Unclosed XML tags recovery (A38)
    unclosed_reply = "<tool_call>\n<function=execute_code>\n<parameter=code>import sys; print(sys.version)"
    calls_recovered, clean_recovered = m.smart_extract_tool_calls(unclosed_reply, "", mock_tools)
    check("unclosed tool_call xml recovered", len(calls_recovered) == 1 and calls_recovered[0]["name"] == "execute_code", f"got {calls_recovered}")

    # ------------------------------------------------------------------
    # [unit] enhanced error recovery & upstream reliability
    # ------------------------------------------------------------------
    print("[unit] enhanced error recovery & upstream reliability")

    # Test 7: Strip markdown code fence inside <tool_call>
    md_tc_reply = "<tool_call>\n```json\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"main.py\"}}\n```\n</tool_call>"
    calls_md, _ = m.smart_extract_tool_calls(md_tc_reply, "", mock_tools)
    check("markdown json inside tool_call stripped and parsed", len(calls_md) == 1 and calls_md[0]["name"] == "read_file" and calls_md[0]["arguments"].get("path") == "main.py", f"got {calls_md}")

    # Test 8: execute_code empty args recovers code from python block (A12)
    empty_exec_reply = "Here is the code:\n```python\nimport os\nprint(os.getcwd())\n```\n<tool_call>\n<function=execute_code>\n</function>\n</tool_call>"
    calls_empty_exec, _ = m.smart_extract_tool_calls(empty_exec_reply, "", mock_tools)
    check("execute_code empty args recovers python code block", len(calls_empty_exec) == 1 and "print(os.getcwd())" in calls_empty_exec[0]["arguments"].get("code", ""), f"got {calls_empty_exec}")

    # Test 9: path as single-item list unwrapped to string
    list_path = m.normalize_tool_args("read_file", {"path": ["src/index.ts"]}, "", mock_tools)
    check("path single list unwrapped to string", list_path.get("path") == "src/index.ts", f"got {list_path}")

    # Test 10: search_files missing pattern populated from file_glob or default
    sf_args = m.normalize_tool_args("search_files", {"file_glob": "*.py"}, "", mock_tools)
    check("search_files missing pattern defaults safely", "pattern" in sf_args and sf_args.get("pattern") == "", f"got {sf_args}")

    # Test 11: is_upstream_busy detects all busy variations
    check("is_upstream_busy helper exists", hasattr(m, "is_upstream_busy"))
    if hasattr(m, "is_upstream_busy"):
        for busy_str in ["服务器繁忙", "系统繁忙", "服务繁忙", "请稍后再试", "请稍后重试"]:
            check(f"is_upstream_busy detects '{busy_str}'", m.is_upstream_busy(f"Error: {busy_str}!"), f"failed for {busy_str}")

    # ------------------------------------------------------------------
    # [unit] smart context & memory retention
    # ------------------------------------------------------------------
    print("[unit] smart context & memory retention")

    # Test 12: System prompt up to 6000 chars is preserved in build_agent_prompt
    long_sys_prompt = "You are Hermes Senior Assistant.\n" + ("Specialized instructions line.\n" * 150)
    msg_with_sys = [
        {"role": "system", "content": long_sys_prompt},
        {"role": "user", "content": "Halo apa kabar?"}
    ]
    prompt_out = m.build_agent_prompt(msg_with_sys, tools=mock_tools)
    check("rich system prompt preserved (not dropped when >1500 chars)", "Specialized instructions line." in prompt_out, f"len prompt={len(prompt_out)}")

    # Test 13: Assistant action in history records tool arguments
    msg_with_action = [
        {"role": "user", "content": "Baca file config"},
        {"role": "assistant", "tool_calls": [{"id": "call_1", "function": {"name": "read_file", "arguments": "{\"path\": \"config.json\"}"}}]},
        {"role": "tool", "name": "read_file", "content": "{\"port\": 8080}"}
    ]
    prompt_action = m.build_agent_prompt(msg_with_action, tools=mock_tools)
    check("assistant action preserves tool arguments", "read_file(path='config.json')" in prompt_action or "read_file(path=\"config.json\")" in prompt_action, f"got {prompt_action}")

    # Test 14: sanitize_observation allows custom larger limit for code/files
    big_code_output = "line_of_code\n" * 300  # ~3900 chars
    sanitized_default = m.sanitize_observation(big_code_output, max_chars=8000)
    check("sanitize_observation retains code up to 8000 chars", len(sanitized_default) == len(big_code_output), f"length={len(sanitized_default)}")

    # Test 15: History retention keeps up to 16 turns
    many_turns = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(20)]
    prompt_turns = m.build_agent_prompt(many_turns, tools=mock_tools)
    check("history retains 16 recent turns", "turn 14" in prompt_turns and "turn 5" in prompt_turns, f"got {prompt_turns}")

    # ------------------------------------------------------------------
    # [unit] model matrix & dynamic discovery
    # ------------------------------------------------------------------
    print("[unit] model matrix & dynamic discovery")

    # Test 16: Ultraspeed model alias exists
    check("MODEL_ALIASES maps ultraspeed", "mimo-v2.6-pro-ultraspeed" in m.MODEL_ALIASES and m.MODEL_ALIASES["mimo-v2.6-pro-ultraspeed"] == "mimo-v2.6-pro-ultraspeed-studio")
    check("MODEL_CATALOG contains ultraspeed", any(cat["id"] in ("mimo-v2.6-pro-ultraspeed", "mimo-v2.6-pro-ultraspeed-studio") for cat in m.MODEL_CATALOG))

    # Test 17: sync_upstream_models helper exists and handles config payload
    check("sync_upstream_models helper exists", hasattr(m, "sync_upstream_models"))
    if hasattr(m, "sync_upstream_models"):
        mock_config = {
            "modelConfigList": [
                {"model": "mimo-v3-future", "name": "MiMo-V3-Future", "enIntro": "Next gen model"}
            ]
        }
        added = m.sync_upstream_models(mock_config)
        check("sync_upstream_models registers new model into catalog", any(cat["id"] == "mimo-v3-future" for cat in m.MODEL_CATALOG))
        check("sync_upstream_models registers new model into aliases", "mimo-v3-future" in m.MODEL_ALIASES)

    # ------------------------------------------------------------------
    # [unit] vision & multimodal upload pipeline
    # ------------------------------------------------------------------
    print("[unit] vision & multimodal upload pipeline")

    check("extract_images_from_messages helper exists", hasattr(m, "extract_images_from_messages"))
    if hasattr(m, "extract_images_from_messages"):
        sample_multimodal = [
            {"role": "user", "content": [
                {"type": "text", "text": "Apa ini?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="}}
            ]}
        ]
        imgs = m.extract_images_from_messages(sample_multimodal)
        check("extracted base64 image", len(imgs) == 1 and imgs[0]["mime_type"] == "image/png", f"got {imgs}")

    check("upload_media_to_mimo helper exists", hasattr(m, "upload_media_to_mimo"))
    check("prepare_multimedias_for_request helper exists", hasattr(m, "prepare_multimedias_for_request"))

    # ------------------------------------------------------------------
    # [unit] anthropic messages API compatibility (/v1/messages)
    # ------------------------------------------------------------------
    print("[unit] anthropic messages API compatibility")

    check("convert_anthropic_request helper exists", hasattr(m, "convert_anthropic_request"))
    check("convert_openai_to_anthropic_response helper exists", hasattr(m, "convert_openai_to_anthropic_response"))

    if hasattr(m, "convert_anthropic_request"):
        ant_req = {
            "model": "claude-3-5-sonnet",
            "system": "You are a coding assistant.",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Cari file"}]}
            ],
            "tools": [
                {
                    "name": "search_files",
                    "description": "Find files",
                    "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}}
                }
            ]
        }
        converted_req = m.convert_anthropic_request(ant_req)
        check("anthropic model mapped", converted_req.get("model") in ("claude-3-5-sonnet", "mimo-v2.6-pro"), f"got {converted_req}")
        check("system message injected", converted_req["messages"][0]["role"] == "system" and "coding assistant" in converted_req["messages"][0]["content"])
        check("anthropic tool converted to openai function", converted_req.get("tools") and converted_req["tools"][0]["function"]["name"] == "search_files")

    if hasattr(m, "convert_openai_to_anthropic_response"):
        openai_mock_resp = {
            "id": "chatcmpl-123",
            "model": "mimo-v2.6-pro",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "File ditemukan.",
                    "reasoning_content": "User meminta mencari file.",
                    "tool_calls": [{
                        "id": "call_abc",
                        "function": {"name": "search_files", "arguments": "{\"pattern\": \"*.py\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }
        ant_resp = m.convert_openai_to_anthropic_response(openai_mock_resp, "claude-3-5-sonnet")
        check("anthropic response structure", ant_resp.get("type") == "message" and ant_resp.get("stop_reason") == "tool_use", f"got {ant_resp}")
        types = [b.get("type") for b in ant_resp.get("content", [])]
        check("anthropic has thinking and tool_use blocks", "thinking" in types and "tool_use" in types, f"got types {types}")

    # Test 18: get_upstream_endpoints routes ultraspeed to /fastchat
    check("get_upstream_endpoints helper exists", hasattr(m, "get_upstream_endpoints"))
    if hasattr(m, "get_upstream_endpoints"):
        u_norm, s_norm = m.get_upstream_endpoints("mimo-v2.6-pro", "dummy_ph")
        check("standard model uses root open-apis", "/fastchat" not in u_norm and "/open-apis/bot/chat" in u_norm)
        u_ultra, s_ultra = m.get_upstream_endpoints("mimo-v2.6-pro-ultraspeed-studio", "dummy_ph")
        check("ultraspeed routes to /fastchat cluster", "/fastchat/open-apis/bot/chat" in u_ultra and "/fastchat/open-apis/chat/conversation/save" in s_ultra)

    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILED -> {FAILURES}")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
