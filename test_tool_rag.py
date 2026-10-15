import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Ensure current dir is in sys.path for direct execution
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimoly_rag import ToolRetriever


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


if __name__ == "__main__":
    test_funcs = [
        test_working_set_preserves_anchor_tools,
        test_working_set_retains_recent_tools,
        test_bm25_ranking_precision,
        test_empty_and_fallback,
        test_ranking_latency_benchmark_300_tools,
        test_concurrency_thread_safety,
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
