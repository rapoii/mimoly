#!/usr/bin/env python3
"""test_rag_agent_e2e.py - End-to-End Autonomous Agent Verification Suite with 250 Tools.

Simulates a complete multi-turn autonomous coding scenario against Mimoly with
250 realistic mock tools (Playwright, Blender, Git, Docker, Postgres, Redis,
K8s, AWS, Slack, Notion, and core anchor tools).

Verifies:
  1. 250 realistic mock tools catalog generation and schema compliance.
  2. Turn 1 (Action Imperative): Prompt assembly under budget, A11 argument healing
     (write_file missing content recovered from markdown code blocks), tool call extracted.
  3. Turn 2 (Tool Output Follow-up & Interception): Workflow in-flight, previous tool
     retention, interception of prose evasion (NUDGE_PROSE), successful auto-nudge retry.
  4. Turn 3 (Final Completion Pass-Through): Legitimate final report after tool execution
     passes cleanly as Verdict.PASS with finish_reason='stop'.
  5. Turn 4 (Out-of-Working-Set Tool Validation): Invocation of an unselected MCP tool
     validates cleanly against the 250-tool global registry via dual-index validation.
  6. Turn 5 (Reasoning Stall Interception): Upstream emitting only <think> without text
     or tool calls intercepted as NUDGE_REASONING, retried with steering directive.
  7. Turn 6 (Conversational Greeting / Question): Pure question/greeting passes through
     cleanly as Verdict.PASS without forced tool calls.
  8. Full multi-turn autonomous coding scenario (Turn 1 -> Turn 2 -> Turn 3).
  9. Concurrency & multi-client session isolation (10+ parallel requests).

Run:
    python projects/mimoly/test_rag_agent_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx
from httpx import ASGITransport

# Ensure current directory is in sys.path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import re
import mimoly
import mimoly_reliability
from mimoly_interceptor import ActionEvaluator, Verdict
from mimoly_rag import ToolRetriever, render_adaptive_tool_catalogue

# Patch wait_throttle to avoid artificial 2.0s delay during fast in-memory tests
async def _noop_wait_throttle(self: Any, min_interval: float = 2.0) -> None:
    pass

mimoly_reliability.PooledAccount.wait_throttle = _noop_wait_throttle


# ---------------------------------------------------------------------------
# 250 Realistic Mock Tools Catalog Generator
# ---------------------------------------------------------------------------

def build_250_mock_tools() -> List[Dict[str, Any]]:
    """Generate 250 realistic mock tools covering anchors and diverse MCP domains."""
    anchors = [
        {
            "name": "terminal",
            "description": "Execute shell commands in bash environment. Returns output and exit code.",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute"},
                "timeout": {"type": "integer", "description": "Max execution time in seconds"},
                "workdir": {"type": "string", "description": "Working directory path"}
            },
            "required": ["command"]
        },
        {
            "name": "read_file",
            "description": "Read text file with line numbers and pagination.",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "offset": {"type": "integer", "description": "Starting line number (1-indexed)"},
                "limit": {"type": "integer", "description": "Maximum lines to read"}
            },
            "required": ["path"]
        },
        {
            "name": "write_file",
            "description": "Write complete content to a file, replacing existing content.",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Complete text content to write"}
            },
            "required": ["path", "content"]
        },
        {
            "name": "patch",
            "description": "Targeted find-and-replace edits in files.",
            "properties": {
                "path": {"type": "string", "description": "File path to edit"},
                "old_string": {"type": "string", "description": "Exact text to find"},
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all matches"}
            },
            "required": ["path", "old_string", "new_string"]
        },
        {
            "name": "search_files",
            "description": "Search file contents (regex) or find files by name glob.",
            "properties": {
                "pattern": {"type": "string", "description": "Search pattern or regex"},
                "target": {"type": "string", "description": "'content' or 'files'"},
                "path": {"type": "string", "description": "Directory to search"},
                "file_glob": {"type": "string", "description": "Optional file filter"}
            },
            "required": ["pattern"]
        },
        {
            "name": "execute_code",
            "description": "Run Python code in persistent interactive session kernel.",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "reset": {"type": "boolean", "description": "Reset session state before running"}
            },
            "required": ["code"]
        },
        {
            "name": "clarify",
            "description": "Ask targeted clarifying questions to the user.",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
                    "description": "List of clarification questions"
                }
            },
            "required": ["questions"]
        }
    ]

    # Domain MCP suites: 243 tools across 10 distinct service domains
    # 7 anchors + 243 MCP tools = exactly 250 tools.
    domains = [
        ("playwright", [
            "browser_navigate", "browser_click", "browser_type", "browser_hover", "browser_press_key",
            "browser_snapshot", "browser_screenshot", "browser_close", "browser_evaluate", "browser_fill_form",
            "browser_select_option", "browser_wait_for", "browser_drag", "browser_drop", "browser_file_upload",
            "browser_handle_dialog", "browser_tabs", "browser_resize", "browser_network_requests", "browser_network_request",
            "browser_console_messages", "browser_run_code_unsafe", "browser_reload", "browser_cookies", "browser_set_cookie",
            "browser_clear_cookies", "browser_pdf", "browser_emulate_device", "browser_get_content", "browser_set_viewport"
        ]),  # 30
        ("blender", [
            "create_object", "add_mesh_cube", "add_mesh_sphere", "render_mesh", "render_scene",
            "render_animation", "set_camera_property", "create_material", "assign_material", "extrude_faces",
            "subdivide_mesh", "smart_uv_project", "add_light", "delete_object", "duplicate_object",
            "set_transform", "apply_modifiers", "add_bone", "create_armature", "parent_mesh",
            "export_gltf", "export_fbx", "import_obj", "bake_textures", "set_render_engine",
            "set_resolution", "add_particle_system", "set_cloth_sim", "remesh_surface", "shade_smooth"
        ]),  # 30
        ("git", [
            "status", "diff", "commit", "log", "checkout",
            "branch_create", "branch_delete", "merge", "rebase", "push",
            "pull", "fetch", "stash_push", "stash_pop", "tag_create",
            "tag_list", "remote_add", "remote_list", "reset_soft", "reset_hard",
            "cherry_pick", "blame", "show", "clean", "clone"
        ]),  # 25
        ("docker", [
            "container_list", "container_run", "container_stop", "container_restart", "container_logs",
            "container_remove", "container_exec", "image_build", "image_pull", "image_push",
            "image_list", "image_remove", "volume_create", "volume_list", "network_create",
            "network_list", "compose_up", "compose_down", "stats", "system_prune",
            "inspect_container", "inspect_image", "top_processes", "copy_file", "commit_container"
        ]),  # 25
        ("postgres", [
            "sql_query", "table_schema", "list_tables", "create_table", "drop_table",
            "add_column", "create_index", "drop_index", "explain_query", "execute_transaction",
            "vacuum_analyze", "get_foreign_keys", "list_indexes", "backup_database", "restore_database",
            "connection_pool_status", "kill_idle_queries", "lock_status", "table_sizes", "row_counts",
            "create_role", "grant_privileges", "revoke_privileges", "server_version", "replication_status"
        ]),  # 25
        ("redis", [
            "get", "set", "del", "exists", "expire",
            "ttl", "keys", "hget", "hset", "hgetall",
            "lpush", "rpush", "lpop", "rpop", "lrange",
            "sadd", "smembers", "zadd", "zrange", "flushdb"
        ]),  # 20
        ("k8s", [
            "get_pods", "describe_pod", "delete_pod", "get_logs", "exec_in_pod",
            "get_deployments", "scale_deployment", "restart_deployment", "get_services", "get_ingresses",
            "get_configmaps", "get_secrets", "apply_manifest", "delete_manifest", "get_nodes",
            "cordon_node", "drain_node", "get_events", "top_pods", "top_nodes",
            "port_forward", "rollout_status", "rollout_undo", "get_namespaces", "create_namespace"
        ]),  # 25
        ("aws", [
            "s3_upload", "s3_download", "s3_list_objects", "s3_delete_object", "lambda_invoke",
            "lambda_update_code", "ec2_describe_instances", "ec2_start_instance", "ec2_stop_instance", "dynamodb_query",
            "dynamodb_put_item", "sqs_send_message", "sqs_receive_message", "sns_publish", "cloudwatch_get_metrics",
            "cloudwatch_tail_logs", "secrets_get_secret", "ecs_list_tasks", "ecs_run_task", "iam_list_users",
            "iam_create_access_key", "route53_list_records", "api_gateway_deploy", "rds_describe_db_instances", "cognito_admin_get_user"
        ]),  # 25
        ("slack", [
            "send_message", "update_message", "delete_message", "read_channel_history", "list_channels",
            "create_channel", "invite_to_channel", "upload_file", "get_user_profile", "add_reaction",
            "remove_reaction", "set_topic", "list_users", "get_presence", "schedule_message",
            "send_ephemeral", "list_pinned_items", "pin_item", "unpin_item", "open_dialog"
        ]),  # 20
        ("notion", [
            "search_pages", "get_page", "create_page", "update_page", "archive_page",
            "get_block_children", "append_block_children", "delete_block", "query_database", "create_database",
            "update_database", "get_user", "list_users", "create_comment", "list_comments",
            "search_by_title", "duplicate_page", "export_page_markdown"
        ])   # 18
    ]

    tools: List[Dict[str, Any]] = []

    for a in anchors:
        tools.append({
            "type": "function",
            "function": {
                "name": a["name"],
                "description": a["description"],
                "parameters": {
                    "type": "object",
                    "properties": a["properties"],
                    "required": a["required"]
                }
            }
        })

    for domain, action_list in domains:
        for act in action_list:
            tool_name = f"mcp__{domain}__{act}"
            desc = f"Execute {act.replace('_', ' ')} in {domain} integration service"
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {"type": "string", "description": "Command payload or target resource identifier"},
                            "options": {"type": "object", "description": "Optional execution parameters"}
                        },
                        "required": ["payload"]
                    }
                }
            })

    assert len(tools) == 250, f"Expected 250 tools, got {len(tools)}"
    return tools


# ---------------------------------------------------------------------------
# Mock Upstream HTTP Infrastructure
# ---------------------------------------------------------------------------

class MockUpstreamResponse:
    """Mock HTTP response providing SSE line iteration and status."""

    def __init__(self, lines: Optional[List[str]] = None, status_code: int = 200):
        self.status_code = status_code
        self._lines = lines or []

    async def __aenter__(self) -> MockUpstreamResponse:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""

    def json(self) -> Dict[str, Any]:
        return {"status": "ok"}


class ScriptedMockClient:
    """Mock AsyncClient simulating upstream MiMo replies across attempts."""

    def __init__(self, attempt_responses: List[str]):
        """
        attempt_responses: list of raw response strings (one per upstream attempt).
        """
        self.attempt_responses = attempt_responses
        self.attempts = 0
        self.captured_queries: List[str] = []
        self.is_closed = False

    def stream(self, method: str, url: str, **kwargs: Any) -> MockUpstreamResponse:
        self.attempts += 1
        payload = kwargs.get("json", {})
        query = payload.get("query", "")
        self.captured_queries.append(query)

        idx = min(self.attempts - 1, len(self.attempt_responses) - 1)
        resp_text = self.attempt_responses[idx]

        lines = [
            f"data: {json.dumps({'type': 'text', 'content': resp_text})}\n",
            "data: [DONE]\n"
        ]
        return MockUpstreamResponse(lines=lines, status_code=200)

    async def post(self, url: str, **kwargs: Any) -> MockUpstreamResponse:
        return MockUpstreamResponse(lines=[], status_code=200)

    async def aclose(self) -> None:
        self.is_closed = True


# ---------------------------------------------------------------------------
# Test Functions
# ---------------------------------------------------------------------------

def test_250_mock_tools_catalogue_validity() -> None:
    """Verify 250 tools generator outputs 250 unique, compliant tool definitions."""
    tools = build_250_mock_tools()
    assert len(tools) == 250, f"Expected 250 tools, got {len(tools)}"

    names = [t["function"]["name"] for t in tools]
    unique_names = set(names)
    assert len(unique_names) == 250, f"Found {250 - len(unique_names)} duplicate tool names"

    anchors = ["terminal", "read_file", "write_file", "patch", "search_files", "execute_code", "clarify"]
    for a in anchors:
        assert a in unique_names, f"Anchor tool '{a}' missing from mock catalogue"

    # Schema integrity check
    for t in tools:
        fn = t.get("function", {})
        assert fn.get("name"), "Tool function missing name"
        assert fn.get("description"), f"Tool {fn.get('name')} missing description"
        params = fn.get("parameters", {})
        assert params.get("type") == "object", f"Tool {fn.get('name')} parameters must be object"
        assert "properties" in params, f"Tool {fn.get('name')} parameters missing properties"
        assert isinstance(params.get("required", []), list), f"Tool {fn.get('name')} required must be list"


def test_turn_1_action_imperative_prompt_and_a11_healing() -> None:
    """Turn 1: Prompt assembly under budget, A11 argument healing on write_file, tool extraction."""
    tools = build_250_mock_tools()
    messages = [
        {"role": "user", "content": "Buatkan file calculator.py dengan fungsi add, subtract, multiply, dan divide."}
    ]

    # 1. Verify prompt assembly preserves anchors and stays under budget
    prompt = mimoly.build_agent_prompt(messages, tools=tools, framework="hermes")
    assert "terminal" in prompt, "Anchor 'terminal' must be preserved in prompt"
    assert "read_file" in prompt, "Anchor 'read_file' must be preserved in prompt"
    assert "write_file" in prompt, "Anchor 'write_file' must be preserved in prompt"
    assert len(prompt) < 12000, f"Prompt length {len(prompt)} exceeded budget"

    # 2. Simulate upstream reply with code in markdown block, but tool_call missing 'content' (A11 defect)
    code_content = (
        "def add(a: float, b: float) -> float:\n"
        "    return a + b\n\n"
        "def subtract(a: float, b: float) -> float:\n"
        "    return a - b\n\n"
        "def multiply(a: float, b: float) -> float:\n"
        "    return a * b\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ValueError('Cannot divide by zero')\n"
        "    return a / b"
    )
    upstream_reply = (
        "Berikut implementasi calculator.py:\n"
        "```python\n"
        f"{code_content}\n"
        "```\n"
        "<tool_call>\n"
        "<function=write_file>\n"
        '{"path": "calculator.py"}\n'
        "</tool_call>"
    )

    # Unit verification of A11 argument recovery
    calls, clean_text = mimoly.smart_extract_tool_calls(upstream_reply, messages[0]["content"], available_tools=tools)
    assert len(calls) == 1, f"Expected 1 tool call, got {len(calls)}"
    assert calls[0]["name"] == "write_file", f"Expected write_file, got {calls[0]['name']}"
    assert calls[0]["arguments"].get("path") == "calculator.py"
    recovered_content = calls[0]["arguments"].get("content", "")
    assert "def add" in recovered_content, "A11 failed to recover 'def add'"
    assert "def divide" in recovered_content, "A11 failed to recover 'def divide'"

    # 3. Full E2E execution through mimoly.app
    async def run_e2e() -> None:
        mock_client = ScriptedMockClient([upstream_reply])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client
        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages,
                    "tools": tools,
                    "stream": False
                })
                assert resp.status_code == 200, f"HTTP status {resp.status_code}: {resp.text}"
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "tool_calls", f"Expected finish_reason 'tool_calls', got {choice['finish_reason']}"
                tcs = choice["message"].get("tool_calls", [])
                assert len(tcs) == 1
                fn_args = json.loads(tcs[0]["function"]["arguments"])
                assert fn_args.get("path") == "calculator.py"
                assert "def add" in fn_args.get("content", "")
                assert "def divide" in fn_args.get("content", "")
        finally:
            mimoly.get_shared_client = orig_client

    asyncio.run(run_e2e())


def test_turn_2_tool_output_followup_and_prose_interception_retry() -> None:
    """Turn 2: In-flight workflow, previous tool retention, prose evasion interception, successful retry."""
    tools = build_250_mock_tools()
    messages = [
        {"role": "user", "content": "Buatkan file calculator.py dengan fungsi add, subtract, multiply, dan divide."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "calculator.py"}'}}
        ]},
        {"role": "tool", "name": "write_file", "content": "File calculator.py written successfully (24 lines)"},
        {"role": "user", "content": "Sekarang jalankan pytest untuk memastikan semua modul calculator.py berfungsi dengan benar."}
    ]

    # 1. Verify working set retains write_file from recent dialogue
    retriever = ToolRetriever()
    selected = retriever.select_working_set(tools, messages, max_dynamic_k=5)
    selected_names = [t["function"]["name"] for t in selected]
    assert "write_file" in selected_names, "Recently used 'write_file' must be retained in working set"
    assert "terminal" in selected_names, "Anchor 'terminal' must be preserved"

    # 2. Verify ActionEvaluator intercepts prose evasion under action mandate
    evasion_reply = "Saya akan menjalankan pengujian pytest sekarang untuk memverifikasi hasilnya."
    evaluator = ActionEvaluator()
    verdict = evaluator.evaluate_turn(
        messages=messages,
        tools_present=True,
        reply_text=evasion_reply,
        tool_calls=[],
        framework="hermes"
    )
    assert verdict == Verdict.NUDGE_PROSE, f"Expected NUDGE_PROSE, got {verdict}"

    # 3. Simulate retry yielding tool_call
    retry_reply = (
        "<tool_call>\n"
        "<function=terminal>\n"
        '{"command": "pytest test_calculator.py -v"}\n'
        "</tool_call>"
    )

    # 4. Full E2E execution through mimoly.app with auto-nudge retry
    async def run_e2e() -> None:
        mock_client = ScriptedMockClient([evasion_reply, retry_reply])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client

        # Patch sleep to keep test execution instantaneous
        orig_sleep = asyncio.sleep
        asyncio.sleep = lambda s: orig_sleep(0.001)

        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages,
                    "tools": tools,
                    "stream": False
                })
                assert resp.status_code == 200, f"HTTP status {resp.status_code}: {resp.text}"
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "tool_calls", f"Expected finish_reason 'tool_calls', got {choice['finish_reason']}"
                tcs = choice["message"].get("tool_calls", [])
                assert len(tcs) == 1
                assert tcs[0]["function"]["name"] == "terminal"
                fn_args = json.loads(tcs[0]["function"]["arguments"])
                assert fn_args.get("command") == "pytest test_calculator.py -v"
                # Verified retry occurred
                assert mock_client.attempts == 2, f"Expected 2 attempts, got {mock_client.attempts}"
                assert "<tool_call>" in mock_client.captured_queries[1], "Retry prompt must include steering directive"
        finally:
            asyncio.sleep = orig_sleep
            mimoly.get_shared_client = orig_client

    asyncio.run(run_e2e())


def test_turn_3_final_completion_pass_through() -> None:
    """Turn 3: Post-tool execution summary cleanly passes through with finish_reason='stop'."""
    tools = build_250_mock_tools()
    messages = [
        {"role": "user", "content": "Buatkan file calculator.py."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "write_file", "arguments": '{"path": "calculator.py"}'}}
        ]},
        {"role": "tool", "name": "write_file", "content": "File calculator.py written successfully"},
        {"role": "user", "content": "Jalankan pytest."},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_2", "type": "function", "function": {"name": "terminal", "arguments": '{"command": "pytest"}'}}
        ]},
        {"role": "tool", "name": "terminal", "content": "==================== 4 passed in 0.04s ===================="},
        {"role": "user", "content": "Bagus! Berikan ringkasan akhir status implementasi dan pengujian calculator.py."}
    ]

    final_summary = (
        "Implementasi `calculator.py` telah selesai dan seluruh pengujian lolos 100%. "
        "Modul menyediakan fungsi add, subtract, multiply, dan divide dengan penanganan "
        "pembagian nol yang tepat. Seluruh 4 pengujian unit berhasil diverifikasi tanpa error."
    )

    # 1. Interceptor check: post-tool summary passes without interception
    evaluator = ActionEvaluator()
    verdict = evaluator.evaluate_turn(
        messages=messages,
        tools_present=True,
        reply_text=final_summary,
        tool_calls=[],
        framework="hermes"
    )
    assert verdict == Verdict.PASS, f"Expected Verdict.PASS for legitimate final completion, got {verdict}"

    # 2. Full E2E execution through mimoly.app
    async def run_e2e() -> None:
        mock_client = ScriptedMockClient([final_summary])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client
        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages,
                    "tools": tools,
                    "stream": False
                })
                assert resp.status_code == 200, f"HTTP status {resp.status_code}: {resp.text}"
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "stop", f"Expected finish_reason 'stop', got {choice['finish_reason']}"
                assert choice["message"].get("tool_calls") is None, "Legitimate completion must not have forced tool calls"
                assert "seluruh pengujian lolos 100%" in choice["message"].get("content", "")
                assert mock_client.attempts == 1, f"Expected 1 attempt (no auto-nudge), got {mock_client.attempts}"
        finally:
            mimoly.get_shared_client = orig_client

    asyncio.run(run_e2e())


def test_turn_4_out_of_working_set_tool_validation() -> None:
    """Turn 4: Invocation of an unselected MCP tool validates cleanly against the 250-tool global registry."""
    tools = build_250_mock_tools()
    messages = [
        {"role": "user", "content": "Periksa semua container docker yang sedang berjalan di sistem."}
    ]

    # Verify that 'mcp__docker__container_list' is in the 250 tools
    tool_names = [t["function"]["name"] for t in tools]
    assert "mcp__docker__container_list" in tool_names

    # Upstream invokes mcp__docker__container_list
    upstream_reply = (
        "<tool_call>\n"
        "<function=mcp__docker__container_list>\n"
        '{"payload": "running", "options": {"all": false}}\n'
        "</tool_call>"
    )

    # 1. Dual-index validation in parser
    calls, clean = mimoly.smart_extract_tool_calls(upstream_reply, messages[0]["content"], available_tools=tools)
    assert len(calls) == 1, f"Expected 1 call, got {len(calls)}"
    assert calls[0]["name"] == "mcp__docker__container_list"
    assert calls[0]["arguments"].get("payload") == "running"
    assert calls[0]["arguments"].get("options") == {"all": False}

    # 2. Full E2E execution through mimoly.app
    async def run_e2e() -> None:
        mock_client = ScriptedMockClient([upstream_reply])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client
        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages,
                    "tools": tools,
                    "stream": False
                })
                assert resp.status_code == 200, f"HTTP status {resp.status_code}: {resp.text}"
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "tool_calls"
                tcs = choice["message"].get("tool_calls", [])
                assert len(tcs) == 1
                assert tcs[0]["function"]["name"] == "mcp__docker__container_list"
                fn_args = json.loads(tcs[0]["function"]["arguments"])
                assert fn_args.get("payload") == "running"
        finally:
            mimoly.get_shared_client = orig_client

    asyncio.run(run_e2e())


def test_turn_5_reasoning_stall_interception() -> None:
    """Turn 5: Upstream emitting only <think> without text or tool calls intercepted as NUDGE_REASONING."""
    tools = build_250_mock_tools()
    messages = [
        {"role": "user", "content": "Jalankan migrasi database sekarang."}
    ]

    # Reasoning-only reply
    stall_reply = (
        "<think>\n"
        "Menganalisis skema database postgresql...\n"
        "Tabel users perlu penambahan kolom status...\n"
        "Mempersiapkan transaksi migration...\n"
        "</think>"
    )

    # 1. ActionEvaluator interceptor evaluation
    evaluator = ActionEvaluator()
    verdict = evaluator.evaluate_turn(
        messages=messages,
        tools_present=True,
        reply_text=stall_reply,
        tool_calls=[],
        framework="hermes"
    )
    assert verdict == Verdict.NUDGE_REASONING, f"Expected NUDGE_REASONING, got {verdict}"

    # 2. Retry emission
    retry_reply = (
        "<think>Melakukan eksekusi migrasi.</think>\n"
        "<tool_call>\n"
        "<function=mcp__postgres__sql_query>\n"
        '{"payload": "ALTER TABLE users ADD COLUMN IF NOT EXISTS status VARCHAR(50);"}\n'
        "</tool_call>"
    )

    # 3. Full E2E execution through mimoly.app
    async def run_e2e() -> None:
        mock_client = ScriptedMockClient([stall_reply, retry_reply])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client

        orig_sleep = asyncio.sleep
        asyncio.sleep = lambda s: orig_sleep(0.001)

        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages,
                    "tools": tools,
                    "stream": False
                })
                assert resp.status_code == 200, f"HTTP status {resp.status_code}: {resp.text}"
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "tool_calls"
                tcs = choice["message"].get("tool_calls", [])
                assert len(tcs) == 1
                assert tcs[0]["function"]["name"] == "mcp__postgres__sql_query"
                assert mock_client.attempts == 2, f"Expected 2 attempts for reasoning stall auto-nudge, got {mock_client.attempts}"
        finally:
            asyncio.sleep = orig_sleep
            mimoly.get_shared_client = orig_client

    asyncio.run(run_e2e())


def test_turn_6_conversational_greeting_and_question_pass_through() -> None:
    """Turn 6: Pure question/greeting passes through cleanly as Verdict.PASS with finish_reason='stop'."""
    tools = build_250_mock_tools()

    # Case A: Greeting
    messages_greet = [
        {"role": "user", "content": "Halo, selamat pagi! Bagaimana kabarmu hari ini?"}
    ]
    greet_reply = "Halo! Selamat pagi. Saya siap membantu Anda dalam proyek software engineering hari ini."

    evaluator = ActionEvaluator()
    v_greet = evaluator.evaluate_turn(messages_greet, True, greet_reply, [], framework="hermes")
    assert v_greet == Verdict.PASS, f"Expected Verdict.PASS for greeting, got {v_greet}"

    # Case B: Conceptual / Informational Query
    messages_query = [
        {"role": "user", "content": "Jelaskan apa itu polymorphism dalam object-oriented programming."}
    ]
    query_reply = (
        "Polymorphism adalah salah satu pilar OOP yang memungkinkan objek dari kelas turunan yang berbeda "
        "untuk merespons interface atau metode yang sama dengan implementasi spesifik masing-masing."
    )
    v_query = evaluator.evaluate_turn(messages_query, True, query_reply, [], framework="hermes")
    assert v_query == Verdict.PASS, f"Expected Verdict.PASS for conceptual query, got {v_query}"

    # Full E2E execution through mimoly.app
    async def run_e2e() -> None:
        mock_client = ScriptedMockClient([greet_reply])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client
        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages_greet,
                    "tools": tools,
                    "stream": False
                })
                assert resp.status_code == 200
                data = resp.json()
                choice = data["choices"][0]
                assert choice["finish_reason"] == "stop"
                assert choice["message"].get("tool_calls") is None
                assert mock_client.attempts == 1
        finally:
            mimoly.get_shared_client = orig_client

    asyncio.run(run_e2e())


def test_turn_7_streaming_sse_interceptor_and_tool_call() -> None:
    """Turn 7: Streaming SSE mode with 250 tools, prose evasion intercepted, retry emits tool call chunks."""
    tools = build_250_mock_tools()
    messages = [
        {"role": "user", "content": "Jalankan pytest -v untuk mengetes seluruh modul."}
    ]

    evasion_attempt = "Saya akan segera menjalankan pytest -v sekarang."
    tool_attempt = (
        "<tool_call>\n"
        "<function=terminal>\n"
        '{"command": "pytest -v"}\n'
        "</tool_call>"
    )

    async def run_streaming() -> None:
        mock_client = ScriptedMockClient([evasion_attempt, tool_attempt])
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock_client

        orig_sleep = asyncio.sleep
        asyncio.sleep = lambda s: orig_sleep(0.001)

        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                async with client.stream("POST", "/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": messages,
                    "tools": tools,
                    "stream": True
                }) as resp:
                    assert resp.status_code == 200
                    chunks = []
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            chunks.append(json.loads(line[6:]))

            assert mock_client.attempts == 2, f"Expected 2 attempts (1 nudge), got {mock_client.attempts}"
            tool_chunks = [c for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
            assert len(tool_chunks) >= 1, "Expected tool_calls chunks in SSE stream"
            tc = tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
            assert tc["function"]["name"] == "terminal"
            args = json.loads(tc["function"]["arguments"])
            assert args["command"] == "pytest -v"

            finish_chunks = [c for c in chunks if c["choices"][0].get("finish_reason") == "tool_calls"]
            assert len(finish_chunks) >= 1, "Expected finish_reason 'tool_calls' in stream"
        finally:
            mimoly.get_shared_client = orig_client
            asyncio.sleep = orig_sleep

    asyncio.run(run_streaming())


def test_full_multi_turn_e2e_session_flow() -> None:
    """Complete 3-turn autonomous coding lifecycle: Create file -> Run tests -> Final summary."""
    tools = build_250_mock_tools()
    session_messages: List[Dict[str, Any]] = []

    # Turn 1: Build file
    user_turn_1 = "Buatkan file calculator.py dengan fungsi add, subtract, multiply, dan divide."
    session_messages.append({"role": "user", "content": user_turn_1})

    reply_turn_1 = (
        "Saya akan membuatkan file calculator.py:\n"
        "```python\n"
        "def add(a, b): return a + b\n"
        "def subtract(a, b): return a - b\n"
        "def multiply(a, b): return a * b\n"
        "def divide(a, b):\n"
        "    if b == 0: raise ValueError('Zero division')\n"
        "    return a / b\n"
        "```\n"
        "<tool_call>\n"
        "<function=write_file>\n"
        '{"path": "calculator.py"}\n'
        "</tool_call>"
    )

    # Turn 2: Run tests (with 1 prose evasion nudge + retry)
    user_turn_2 = "Jalankan pytest sekarang untuk menguji modul tersebut."
    evasion_turn_2 = "Saya akan menjalankan test sekarang dengan pytest."
    retry_turn_2 = (
        "<tool_call>\n"
        "<function=terminal>\n"
        '{"command": "pytest test_calculator.py"}\n'
        "</tool_call>"
    )

    # Turn 3: Final report
    user_turn_3 = "Bagus! Berikan ringkasan akhir."
    summary_turn_3 = (
        "Implementasi dan pengujian calculator.py telah selesai sepenuhnya. "
        "Seluruh fungsi bekerja sesuai spesifikasi dan lulus seluruh unit test."
    )

    async def execute_session() -> None:
        orig_client = mimoly.get_shared_client
        orig_sleep = asyncio.sleep
        asyncio.sleep = lambda s: orig_sleep(0.001)

        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                # --- TURN 1 ---
                mock_1 = ScriptedMockClient([reply_turn_1])
                mimoly.get_shared_client = lambda: mock_1
                r1 = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": session_messages,
                    "tools": tools,
                    "stream": False
                })
                assert r1.status_code == 200
                d1 = r1.json()["choices"][0]
                assert d1["finish_reason"] == "tool_calls"
                tc1 = d1["message"]["tool_calls"][0]
                assert tc1["function"]["name"] == "write_file"
                args1 = json.loads(tc1["function"]["arguments"])
                assert args1.get("path") == "calculator.py"
                assert "def add" in args1.get("content", "")

                # Record Turn 1 in dialogue history
                session_messages.append(d1["message"])
                session_messages.append({
                    "role": "tool",
                    "name": "write_file",
                    "tool_call_id": tc1["id"],
                    "content": "File calculator.py written successfully"
                })

                # --- TURN 2 ---
                session_messages.append({"role": "user", "content": user_turn_2})
                mock_2 = ScriptedMockClient([evasion_turn_2, retry_turn_2])
                mimoly.get_shared_client = lambda: mock_2
                r2 = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": session_messages,
                    "tools": tools,
                    "stream": False
                })
                assert r2.status_code == 200
                d2 = r2.json()["choices"][0]
                assert d2["finish_reason"] == "tool_calls"
                assert mock_2.attempts == 2, "Prose evasion must trigger auto-nudge retry"
                tc2 = d2["message"]["tool_calls"][0]
                assert tc2["function"]["name"] == "terminal"

                # Record Turn 2 in dialogue history
                session_messages.append(d2["message"])
                session_messages.append({
                    "role": "tool",
                    "name": "terminal",
                    "tool_call_id": tc2["id"],
                    "content": "==== 4 passed in 0.04s ===="
                })

                # --- TURN 3 ---
                session_messages.append({"role": "user", "content": user_turn_3})
                mock_3 = ScriptedMockClient([summary_turn_3])
                mimoly.get_shared_client = lambda: mock_3
                r3 = await client.post("/v1/chat/completions", json={
                    "model": "mimo-v2.6-pro",
                    "messages": session_messages,
                    "tools": tools,
                    "stream": False
                })
                assert r3.status_code == 200
                d3 = r3.json()["choices"][0]
                assert d3["finish_reason"] == "stop"
                assert d3["message"].get("tool_calls") is None
                assert "telah selesai sepenuhnya" in d3["message"].get("content", "")
                assert mock_3.attempts == 1

        finally:
            asyncio.sleep = orig_sleep
            mimoly.get_shared_client = orig_client

    asyncio.run(execute_session())


def test_concurrency_multi_client_session_isolation() -> None:
    """Verify zero state leakage across 10 concurrent agent client sessions."""
    tools = build_250_mock_tools()

    class ConcurrentDispatchMockClient:
        def __init__(self) -> None:
            self.is_closed = False

        def stream(self, method: str, url: str, **kwargs: Any) -> MockUpstreamResponse:
            payload = kwargs.get("json", {})
            query = payload.get("query", "")
            m = re.search(r"client[_\s]+(\d+)", query)
            idx_str = m.group(1) if m else "0"
            resp_text = (
                f"<tool_call>\n"
                f"<function=terminal>\n"
                f'{{"command": "echo client_{idx_str}"}}\n'
                f"</tool_call>"
            )
            lines = [
                f"data: {json.dumps({'type': 'text', 'content': resp_text})}\n",
                "data: [DONE]\n"
            ]
            return MockUpstreamResponse(lines=lines, status_code=200)

        async def post(self, url: str, **kwargs: Any) -> MockUpstreamResponse:
            return MockUpstreamResponse(lines=[], status_code=200)

        async def aclose(self) -> None:
            self.is_closed = True

    async def single_client(idx: int, client: httpx.AsyncClient) -> None:
        resp = await client.post("/v1/chat/completions", json={
            "model": "mimo-v2.6-pro",
            "messages": [{"role": "user", "content": f"Jalankan command test client {idx}"}],
            "tools": tools,
            "stream": False
        })
        assert resp.status_code == 200
        data = resp.json()
        tcs = data["choices"][0]["message"].get("tool_calls", [])
        assert len(tcs) == 1
        args = json.loads(tcs[0]["function"]["arguments"])
        assert f"echo client_{idx}" in args.get("command", "")

    async def run_concurrent() -> None:
        mock = ConcurrentDispatchMockClient()
        orig_client = mimoly.get_shared_client
        mimoly.get_shared_client = lambda: mock
        try:
            transport = ASGITransport(app=mimoly.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                tasks = [single_client(i, client) for i in range(10)]
                await asyncio.gather(*tasks)
        finally:
            mimoly.get_shared_client = orig_client

    asyncio.run(run_concurrent())


# ---------------------------------------------------------------------------
# CLI Test Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("======================================================================")
    print("Mimoly E2E Multi-Turn Simulation with 250 Tools")
    print("======================================================================\n")

    test_suite = [
        ("Catalog Validity (250 Tools)", test_250_mock_tools_catalogue_validity),
        ("Turn 1: Action Imperative & A11 Healing", test_turn_1_action_imperative_prompt_and_a11_healing),
        ("Turn 2: Follow-up & Prose Interception Retry", test_turn_2_tool_output_followup_and_prose_interception_retry),
        ("Turn 3: Final Completion Pass-Through", test_turn_3_final_completion_pass_through),
        ("Turn 4: Out-of-Working-Set Tool Validation", test_turn_4_out_of_working_set_tool_validation),
        ("Turn 5: Reasoning Stall Interception", test_turn_5_reasoning_stall_interception),
        ("Turn 6: Conversational Greeting / Question", test_turn_6_conversational_greeting_and_question_pass_through),
        ("Turn 7: Streaming SSE Interceptor & Tool Call", test_turn_7_streaming_sse_interceptor_and_tool_call),
        ("Multi-Turn Full Session Flow (Turns 1-3)", test_full_multi_turn_e2e_session_flow),
        ("Concurrency Multi-Client Session Isolation", test_concurrency_multi_client_session_isolation),
    ]

    passed = 0
    failed = 0
    failures: List[str] = []

    for name, fn in test_suite:
        t0 = time.perf_counter()
        try:
            fn()
            dt = (time.perf_counter() - t0) * 1000
            print(f"  PASS  {name} ({dt:.1f}ms)")
            passed += 1
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000
            print(f"  FAIL  {name} ({dt:.1f}ms) -> {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failures.append(name)

    print("\n----------------------------------------------------------------------")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("----------------------------------------------------------------------")

    sys.exit(0 if failed == 0 else 1)
