from offline_test_support import install_network_deny
install_network_deny()

import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Ensure current dir is in sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimoly_rag import ToolRetriever, render_adaptive_tool_catalogue


def test_working_set_preserves_anchor_tools():
    all_tools = [
        {"function": {"name": "terminal", "description": "Run shell commands"}},
        {"function": {"name": "read_file", "description": "Read file"}},
        {"function": {"name": "write_file", "description": "Write file"}},
        {"function": {"name": "patch", "description": "Patch file"}},
        {"function": {"name": "search_files", "description": "Search files"}},
        {"function": {"name": "mcp__playwright__browser_click", "description": "Click element"}},
        {"function": {"name": "mcp__playwright__browser_navigate", "description": "Navigate URL"}},
    ]
    messages = [{"role": "user", "content": "Buka google.com di browser"}]
    retriever = ToolRetriever()
    selected = retriever.select_working_set(all_tools, messages, max_dynamic_k=2)
    names = [t["function"]["name"] for t in selected]
    for anchor in ["terminal", "read_file", "write_file", "patch", "search_files"]:
        assert anchor in names, f"Anchor tool '{anchor}' missing from selected working set"
    assert "mcp__playwright__browser_navigate" in names, "Expected browser_navigate to be selected"


def test_working_set_retains_recent_tools():
    all_tools = [
        {"function": {"name": "terminal", "description": "Run shell commands"}},
        {"function": {"name": "read_file", "description": "Read file"}},
        {"function": {"name": "mcp__db__query_postgres", "description": "Execute SQL queries on PostgreSQL"}},
        {"function": {"name": "mcp__blender__render_mesh", "description": "Render 3D mesh in Blender"}},
        {"function": {"name": "mcp__playwright__browser_navigate", "description": "Navigate URL in browser"}},
    ]
    # Dialogue where assistant previously used query_postgres
    messages = [
        {"role": "user", "content": "Query users table"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "mcp__db__query_postgres", "arguments": "{\"query\": \"SELECT * FROM users\"}"}}
        ]},
        {"role": "tool", "name": "mcp__db__query_postgres", "content": "id, name\n1, Alice"},
        {"role": "user", "content": "Now summarize the result"}
    ]
    retriever = ToolRetriever()
    # Dynamic K is 1, user query mentions 'summarize', so blender or playwright wouldn't match,
    # but query_postgres must be preserved via contextual retention tier!
    selected = retriever.select_working_set(all_tools, messages, max_dynamic_k=1)
    names = [t["function"]["name"] for t in selected]
    assert "terminal" in names, "Anchor tool 'terminal' must be preserved"
    assert "read_file" in names, "Anchor tool 'read_file' must be preserved"
    assert "mcp__db__query_postgres" in names, "Recently used tool 'mcp__db__query_postgres' must be retained"


def test_bm25_ranking_precision():
    all_tools = [
        {"function": {"name": "mcp__blender__add_mesh_cube", "description": "Add a 3D mesh cube in Blender scene"}},
        {"function": {"name": "mcp__blender__render_animation", "description": "Render animation frames in Blender"}},
        {"function": {"name": "mcp__playwright__browser_navigate", "description": "Navigate to URL in web browser"}},
        {"function": {"name": "mcp__playwright__browser_click", "description": "Click an element on the webpage"}},
        {"function": {"name": "mcp__db__sql_execute", "description": "Execute raw SQL query on database"}},
    ]
    retriever = ToolRetriever()

    # Query 1: browser navigation
    msgs1 = [{"role": "user", "content": "Tolong buka website https://example.com di browser"}]
    selected1 = retriever.select_working_set(all_tools, msgs1, max_dynamic_k=2)
    names1 = [t["function"]["name"] for t in selected1]
    assert "mcp__playwright__browser_navigate" in names1, "Browser navigate must rank in top 2 for browser URL task"

    # Query 2: 3D render in blender
    msgs2 = [{"role": "user", "content": "Render the 3D scene animation in blender"}]
    selected2 = retriever.select_working_set(all_tools, msgs2, max_dynamic_k=1)
    names2 = [t["function"]["name"] for t in selected2]
    assert "mcp__blender__render_animation" in names2, "Blender animation tool must rank top for 3D render task"


def test_empty_and_fallback():
    retriever = ToolRetriever()
    # Empty tools list
    assert retriever.select_working_set([], [{"role": "user", "content": "hello"}]) == []

    # Tools without anchor tools
    custom_tools = [
        {"function": {"name": "tool_a", "description": "alpha tool"}},
        {"function": {"name": "tool_b", "description": "beta tool"}},
    ]
    selected = retriever.select_working_set(custom_tools, [{"role": "user", "content": "alpha"}], max_dynamic_k=1)
    names = [t["function"]["name"] for t in selected]
    assert "tool_a" in names


def test_ranking_latency_benchmark_300_tools():
    # Synthesize 300 realistic MCP tools
    domains = ["playwright", "blender", "slack", "github", "docker", "postgres", "redis", "k8s", "aws", "notion"]
    actions = ["create", "delete", "update", "get", "list", "search", "export", "import", "build", "run",
               "stop", "start", "restart", "deploy", "check", "verify", "diff", "merge", "pull", "push",
               "query", "fetch", "stream", "record", "convert", "compress", "extract", "validate", "render", "sync"]

    tools = []
    for d in domains:
        for a in actions:
            tools.append({
                "type": "function",
                "function": {
                    "name": f"mcp__{d}__{a}_resource",
                    "description": f"Tool to {a} resource objects in {d} service cluster and environment"
                }
            })
    assert len(tools) == 300

    retriever = ToolRetriever()
    messages = [{"role": "user", "content": "Please deploy and restart the docker container image"}]

    # Warm-up call
    retriever.select_working_set(tools, messages, max_dynamic_k=5)

    # Benchmark 30 calls
    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        selected = retriever.select_working_set(tools, messages, max_dynamic_k=5)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    median_ms = sorted(latencies)[len(latencies) // 2]
    min_ms = min(latencies)
    print(f"\n[latency benchmark] 300 tools: Min={min_ms:.3f}ms, Median={median_ms:.3f}ms, Avg={avg_ms:.3f}ms")

    # Strict constraint: median retrieval & ranking latency <= 2.0 ms
    assert median_ms <= 2.0, f"Ranking latency {median_ms:.3f}ms exceeded 2.0ms budget for 300 tools"
    selected_names = [t["function"]["name"] for t in selected]
    assert any("docker" in n for n in selected_names), "Expected docker tool in selected working set"


def test_class_level_select_working_set():
    all_tools = [
        {"function": {"name": "terminal", "description": "Run shell commands"}},
        {"function": {"name": "read_file", "description": "Read file"}},
        {"function": {"name": "mcp__playwright__browser_navigate", "description": "Navigate URL in browser"}},
    ]
    messages = [{"role": "user", "content": "Buka web https://example.com"}]

    # Direct class-level invocation without manual instantiation
    selected = ToolRetriever.select_working_set(all_tools, messages, max_dynamic_k=1)
    names = [t["function"]["name"] for t in selected]
    assert "terminal" in names, "Anchor tool 'terminal' must be selected via class-level call"
    assert "read_file" in names, "Anchor tool 'read_file' must be selected via class-level call"
    assert "mcp__playwright__browser_navigate" in names, "Dynamic tool must be selected via class-level call"

    # Instance-level call with custom anchor configuration
    custom_retriever = ToolRetriever(anchor_tools={"custom_anchor"})
    custom_tools = [
        {"function": {"name": "custom_anchor", "description": "Custom anchor tool"}},
        {"function": {"name": "terminal", "description": "Standard terminal"}},
    ]
    custom_selected = custom_retriever.select_working_set(custom_tools, messages, max_dynamic_k=0)
    custom_names = [t["function"]["name"] for t in custom_selected]
    assert custom_names == ["custom_anchor"], f"Expected custom anchor only, got {custom_names}"


def test_malformed_and_nameless_tool_filtering():
    malformed_tools = [
        None,
        {},
        {"function": None},
        {"function": {}},
        {"function": {"name": ""}},
        {"function": {"name": None}},
        {"name": ""},
        {"name": None},
        12345,
        "not_a_valid_dict",
        {"function": {"name": "valid_tool", "description": "A valid tool for tests"}},
    ]
    messages = [{"role": "user", "content": "Run tests on valid tool"}]

    # Ensure no malformed tool leaks into selected working set
    selected = ToolRetriever.select_working_set(malformed_tools, messages, max_dynamic_k=5)
    assert len(selected) == 1, f"Expected exactly 1 valid tool, got {len(selected)}: {selected}"
    name = selected[0]["function"]["name"]
    assert name == "valid_tool", f"Expected 'valid_tool', got {name}"

    # All-malformed list must return empty list without crashing
    all_malformed = [None, {}, {"function": {}}, {"name": ""}, "bad_entry"]
    assert ToolRetriever.select_working_set(all_malformed, messages) == []

    # Malformed messages entries must not cause crashes
    malformed_messages = [None, 123, "not_a_dict", {"role": "user", "content": "valid text"}]
    res = ToolRetriever.select_working_set(malformed_tools, malformed_messages)
    assert len(res) == 1 and res[0]["function"]["name"] == "valid_tool"


def test_stable_chronological_retention_order():
    tools = [
        {"function": {"name": f"tool_{k}", "description": f"Tool {k} description"}}
        for k in ["alpha", "beta", "gamma", "delta", "epsilon"]
    ]
    # 3 assistant turns with tool calls in specific order
    messages = [
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "tool_beta"}},
            {"function": {"name": "tool_alpha"}},
        ]},
        {"role": "tool", "name": "tool_beta", "content": "res"},
        {"role": "user", "content": "turn 2"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "tool_gamma"}},
        ]},
        {"role": "user", "content": "turn 3"},
        {"role": "assistant", "tool_calls": [
            {"function": {"name": "tool_beta"}},  # Repeated tool call
            {"function": {"name": "tool_delta"}},
        ]},
    ]
    retriever = ToolRetriever(anchor_tools=set())  # Empty anchors to isolate retention order
    selected = retriever.select_working_set(tools, messages, max_dynamic_k=0)
    selected_names = [t["function"]["name"] for t in selected]

    # Chronological order of first appearance across window: beta -> alpha -> gamma -> delta
    expected = ["tool_beta", "tool_alpha", "tool_gamma", "tool_delta"]
    assert selected_names == expected, f"Order mismatch: expected {expected}, got {selected_names}"


def test_concurrency_thread_safety():
    tools = [
        {"function": {"name": f"tool_{i}", "description": f"Perform action {i} on system"}}
        for i in range(50)
    ]
    retriever = ToolRetriever()

    def worker(i):
        msgs = [{"role": "user", "content": f"Perform action {i % 10} on system"}]
        res = retriever.select_working_set(tools, msgs, max_dynamic_k=3)
        return len(res)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(worker, range(50)))

    assert len(results) == 50
    assert all(r > 0 for r in results)


def test_render_adaptive_tool_catalogue_full_schema():
    tools = [
        {
            "function": {
                "name": "write_file",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Target file path"},
                        "content": {"type": "string", "description": "File content"},
                    },
                    "required": ["path", "content"],
                },
            }
        }
    ]
    rendered = render_adaptive_tool_catalogue(tools, budget_chars=4000, framework="hermes")
    assert "write_file" in rendered
    assert "Target file path" in rendered
    assert "<tool_call>" in rendered
    assert "path" in rendered
    assert "content" in rendered
    assert "string" in rendered


def test_render_adaptive_tool_catalogue_frameworks():
    tools = [
        {
            "function": {
                "name": "read_file",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"],
                },
            }
        }
    ]
    # Hermes / Default -> Qwen XML
    xml_hermes = render_adaptive_tool_catalogue(tools, framework="hermes")
    assert "<tool_call>" in xml_hermes
    assert "<function=read_file>" in xml_hermes
    assert "<parameter=path>" in xml_hermes

    xml_default = render_adaptive_tool_catalogue(tools, framework="default")
    assert "<tool_call>" in xml_default
    assert "<function=read_file>" in xml_default

    # Claude Code -> Anthropic tool_use JSON
    json_claude = render_adaptive_tool_catalogue(tools, framework="claude-code")
    assert '"type": "tool_use"' in json_claude
    assert '"name": "read_file"' in json_claude

    # Codex -> OpenAI function calling JSON
    json_codex = render_adaptive_tool_catalogue(tools, framework="codex")
    assert '"tool_calls"' in json_codex
    assert '"name": "read_file"' in json_codex


def test_budget_pruning_description_truncation():
    # Long param description > 150 chars
    long_desc = "Detailed explanation of the target file path on the filesystem that needs to be accessed. " * 2 + "End."
    tools = [
        {
            "function": {
                "name": "write_file",
                "description": "Write file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": long_desc},
                        "content": {"type": "string", "description": "Content string"},
                    },
                    "required": ["path", "content"],
                },
            }
        }
    ]
    # Render with generous budget -> full description is present
    rendered_full = render_adaptive_tool_catalogue(tools, budget_chars=5000)
    assert long_desc in rendered_full

    # Render with budget smaller than full -> triggers 80-char truncation
    # But generous enough that throttling is not needed
    target_budget = len(rendered_full) - 50
    rendered_pruned = render_adaptive_tool_catalogue(tools, budget_chars=target_budget)
    assert len(rendered_pruned) <= target_budget
    assert long_desc not in rendered_pruned
    # Strictly preserves property names, types, and required status
    assert "path" in rendered_pruned
    assert "content" in rendered_pruned
    assert "string" in rendered_pruned


def test_budget_pruning_tool_description_truncation():
    """Verify that tool top-level description is truncated to 80 chars under budget pressure."""
    long_tool_desc = "This is an extremely verbose tool description detailing comprehensive edge cases and options that extends well beyond eighty characters."
    tools = [
        {
            "function": {
                "name": "deep_inspect",
                "description": long_tool_desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "Target name"},
                    },
                    "required": ["target"],
                },
            }
        }
    ]
    # Full budget -> full description is present
    rendered_full = render_adaptive_tool_catalogue(tools, budget_chars=5000)
    assert long_tool_desc in rendered_full

    # Under budget pressure (Phase 1) -> top-level tool description is truncated to 80 chars
    target_budget = len(rendered_full) - 30
    rendered_pruned = render_adaptive_tool_catalogue(tools, budget_chars=target_budget)
    assert len(rendered_pruned) <= target_budget
    assert long_tool_desc not in rendered_pruned
    assert long_tool_desc[:80] in rendered_pruned
    assert "deep_inspect" in rendered_pruned
    assert "target" in rendered_pruned


def test_render_adaptive_catalogue_flexibility_and_json_literals():
    """Verify budget_chars=None fallback, selected_tools keyword argument, and XML json.dumps literals."""
    tools = [
        {
            "function": {
                "name": "custom_filter",
                "description": "Filter data records",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "is_active": {"type": "boolean", "description": "Active status"},
                        "limit": {"type": "integer", "description": "Maximum count"},
                    },
                    "required": ["is_active", "limit"],
                },
            }
        }
    ]

    # Test selected_tools keyword argument
    rendered_kw = render_adaptive_tool_catalogue(selected_tools=tools)
    assert "custom_filter" in rendered_kw

    # Test budget_chars=None defaults to 4000
    rendered_none = render_adaptive_tool_catalogue(tools, budget_chars=None)
    assert "custom_filter" in rendered_none
    assert len(rendered_none) <= 4000

    # Test XML formatting of non-string dummy arguments with json.dumps
    # Boolean must format as lowercase "true", integer as "1"
    assert "<parameter=is_active>true</parameter>" in rendered_kw
    assert "<parameter=limit>1</parameter>" in rendered_kw


def test_budget_pruning_dynamic_throttling_preserves_anchors():
    anchor_names = ["terminal", "read_file", "write_file", "patch", "search_files"]
    tools = []
    for a in anchor_names:
        tools.append({
            "function": {
                "name": a,
                "description": f"Core anchor tool {a}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg": {"type": "string", "description": f"Arg for {a}"}
                    },
                    "required": ["arg"],
                },
            }
        })

    # Add 5 dynamic tools with substantial schemas
    dynamic_names = [f"mcp__service__tool_{i}" for i in range(5)]
    for d in dynamic_names:
        tools.append({
            "function": {
                "name": d,
                "description": f"Dynamic specialized tool {d}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param_1": {"type": "string", "description": f"Detailed param 1 for {d}"},
                        "param_2": {"type": "integer", "description": f"Detailed param 2 for {d}"},
                    },
                    "required": ["param_1"],
                },
            }
        })

    # Baseline with large budget has all tools
    rendered_all = render_adaptive_tool_catalogue(tools, budget_chars=10000)
    for a in anchor_names:
        assert a in rendered_all
    for d in dynamic_names:
        assert d in rendered_all

    # Constrained budget that forces pruning dynamic tools
    tight_budget = len(rendered_all) - 600
    rendered_pruned = render_adaptive_tool_catalogue(tools, budget_chars=tight_budget)

    # Budget strictly respected
    assert len(rendered_pruned) <= tight_budget

    # ALL anchor tools unconditionally preserved
    for a in anchor_names:
        assert a in rendered_pruned, f"Anchor tool '{a}' was unexpectedly dropped!"

    # Dynamic tools are throttled
    dropped_dynamic = [d for d in dynamic_names if d not in rendered_pruned]
    assert len(dropped_dynamic) > 0, "Expected at least one dynamic tool to be throttled"


def test_render_adaptive_catalogue_edge_cases():
    # Empty tools
    assert render_adaptive_tool_catalogue([]) == ""

    # Malformed / empty tools
    malformed = [None, {}, {"function": {}}, {"name": ""}]
    assert render_adaptive_tool_catalogue(malformed) == ""

    # Tool with empty parameters dict
    tools = [{"function": {"name": "simple_tool", "description": "No params"}}]
    rendered = render_adaptive_tool_catalogue(tools)
    assert "simple_tool" in rendered
    assert "No params" in rendered


def test_render_latency_benchmark():
    tools = [
        {
            "function": {
                "name": f"mcp__service__tool_{i}",
                "description": f"Perform action {i} on the remote target environment",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "param_1": {"type": "string", "description": f"Detailed parameter 1 for action {i}"},
                        "param_2": {"type": "integer", "description": f"Limit counter for operation {i}"},
                    },
                    "required": ["param_1"],
                },
            }
        }
        for i in range(15)
    ]
    # Warmup
    for _ in range(5):
        render_adaptive_tool_catalogue(tools, budget_chars=2000, framework="hermes")

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        res = render_adaptive_tool_catalogue(tools, budget_chars=2000, framework="hermes")
        times.append((time.perf_counter() - t0) * 1000)

    median_ms = sorted(times)[len(times) // 2]
    print(f"\n[render latency] 15 tools with pruning: Min={min(times):.3f}ms, Median={median_ms:.3f}ms, Avg={sum(times)/len(times):.3f}ms")
    assert median_ms < 2.0, f"Render latency too high: {median_ms:.3f}ms > 2.0ms"


if __name__ == "__main__":
    test_funcs = [
        test_working_set_preserves_anchor_tools,
        test_working_set_retains_recent_tools,
        test_bm25_ranking_precision,
        test_empty_and_fallback,
        test_ranking_latency_benchmark_300_tools,
        test_class_level_select_working_set,
        test_malformed_and_nameless_tool_filtering,
        test_stable_chronological_retention_order,
        test_concurrency_thread_safety,
        test_render_adaptive_tool_catalogue_full_schema,
        test_render_adaptive_tool_catalogue_frameworks,
        test_budget_pruning_description_truncation,
        test_budget_pruning_tool_description_truncation,
        test_render_adaptive_catalogue_flexibility_and_json_literals,
        test_budget_pruning_dynamic_throttling_preserves_anchors,
        test_render_adaptive_catalogue_edge_cases,
        test_render_latency_benchmark,
    ]
    passed = 0
    failed = 0
    for fn in test_funcs:
        name = fn.__name__
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}  ({e})")
            failed += 1

    print(f"\nRESULT: {passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
