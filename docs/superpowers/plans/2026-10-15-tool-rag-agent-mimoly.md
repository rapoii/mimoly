# Tool-RAG & Strict Action Interceptor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Mimoly into a 100% reliable AI agent Web2API proxy with zero missed tool calls, in-memory BM25 Tool-RAG, adaptive full schemas, and strict action interception for 1000+ concurrent users.

**Architecture:** A standalone in-memory BM25 working set retriever (`mimoly_rag.py`) curates ~8-12 tools (Anchor + Retention + Top-K Dynamic) from 200+ client tools. Mimoly injects full parameter schemas into prompt assembly, validates parsed calls against the global registry, and intercepts empty `<think>` stalls or prose evasions with bounded framework-aware auto-nudge retries.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, httpx, asyncio, standard library (re, json, math, collections).

**Spec:** `projects/mimoly/docs/superpowers/specs/2026-10-15-tool-rag-agent-mimoly-design.md`

## Global Constraints

- Pure Python in-memory implementation without external vector databases (no Milvus, ChromaDB, or Qdrant).
- Retrieval and ranking latency must not exceed 2.0 ms for 300 tool definitions.
- Zero shared mutable state across requests; fully safe for 1000+ concurrent requests.
- Backward-compatible with existing FastChat / Ultraspeed retry budgets, account rotation, and streaming SSE protocols.
- Preserve existing tests in `test_reliability.py` and `test_upstream_frames.py` at 100% passing state.

## Review Focus

- **Anchor tool omission:** An incoming tool list contains 200+ MCP tools and the user prompt matches none of the anchor tools; all 5 anchor tools must still be unconditionally preserved in the working set.
- **Large MCP schema explosion:** 12 selected tools have verbose descriptions exceeding 4,000 characters; parameter descriptions must be cleanly truncated to 80 chars, and if still over budget, $K$ dynamic tools must be throttled down to ensure parameter names/types never drop.
- **In-flight streaming retry collision:** If upstream starts emitting answer tokens (`_content_started=True`), auto-nudge retry must be locked out to avoid corrupting client SSE parsers.
- **Informational query false positive:** User says *"Halo, apa kabar?"* or *"Jelaskan konsep OOP"*; the interceptor must NOT force a tool call or trigger retries.
- **Out-of-working-set invocation:** Model calls an MCP tool that was omitted from the prompt's working set; `smart_extract_tool_calls` must successfully validate and extract it against the full client tool catalogue.

---

### Task 1: In-Memory BM25 & Working Set Retriever

**Files:**
- Create: `projects/mimoly/mimoly_rag.py`
- Test: `projects/mimoly/test_tool_rag.py`

**Interfaces:**
- Consumes: `tools: List[Dict[str, Any]]`, `messages: List[Dict[str, Any]]`, `max_dynamic_k: int = 5`.
- Produces: `ToolRetriever.select_working_set(tools, messages, max_dynamic_k) -> List[Dict[str, Any]]`.

- [ ] **Step 1: Write the failing test for ToolRetriever**

```python
# projects/mimoly/test_tool_rag.py
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
        assert anchor in names
    assert "mcp__playwright__browser_navigate" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python projects/mimoly/test_tool_rag.py`
Expected: FAIL (ModuleNotFoundError: No module named 'mimoly_rag')

- [ ] **Step 3: Implement `ToolRetriever` in `projects/mimoly/mimoly_rag.py`**

Implement in-memory BM25 indexing over tool names + descriptions, Anchor Tool partitioning, last-turn tool retention, and top-K dynamic selection.

- [ ] **Step 4: Run test to verify it passes**

Run: `python projects/mimoly/test_tool_rag.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/mimoly/mimoly_rag.py projects/mimoly/test_tool_rag.py
git commit -m "feat(rag): add in-memory BM25 working set tool retriever"
```

---

### Task 2: Adaptive Schema & Few-Shot Generator with Budget Pruning

**Files:**
- Modify: `projects/mimoly/mimoly_rag.py`
- Test: `projects/mimoly/test_tool_rag.py`

**Interfaces:**
- Consumes: `selected_tools: List[Dict[str, Any]]`, `budget_chars: int = 4000`, `framework: str = "default"`.
- Produces: `render_adaptive_tool_catalogue(tools, budget_chars, framework) -> str`.

- [ ] **Step 1: Write the failing test for schema budgeting and pruning**

```python
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
                        "content": {"type": "string", "description": "File content"}
                    },
                    "required": ["path", "content"]
                }
            }
        }
    ]
    rendered = render_adaptive_tool_catalogue(tools, budget_chars=4000, framework="hermes")
    assert "write_file" in rendered
    assert "Target file path" in rendered
    assert "<tool_call>" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python projects/mimoly/test_tool_rag.py`
Expected: FAIL (NameError: name 'render_adaptive_tool_catalogue' is not defined)

- [ ] **Step 3: Implement `render_adaptive_tool_catalogue` in `projects/mimoly/mimoly_rag.py`**

Format full parameters, types, required fields, and compact few-shot examples. Apply description shortening to 80 chars if total > budget, and throttle dynamic tools if still over budget.

- [ ] **Step 4: Run test to verify it passes**

Run: `python projects/mimoly/test_tool_rag.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/mimoly/mimoly_rag.py projects/mimoly/test_tool_rag.py
git commit -m "feat(rag): add adaptive full schema rendering with budget-aware pruning"
```

---

### Task 3: Strict Action Interceptor Logic

**Files:**
- Create: `projects/mimoly/mimoly_interceptor.py`
- Test: `projects/mimoly/test_action_interceptor.py`

**Interfaces:**
- Consumes: `messages: List[Dict[str, Any]]`, `tools_present: bool`, `reply_text: str`, `tool_calls: List[Dict[str, Any]]`, `framework: str`.
- Produces: `ActionEvaluator.evaluate_turn(...) -> TurnVerdict (PASS, NUDGE_REASONING, NUDGE_PROSE)`.
- Produces: `ActionEvaluator.get_steering_directive(framework, available_tool_names) -> str`.

- [ ] **Step 1: Write the failing test for ActionEvaluator**

```python
# projects/mimoly/test_action_interceptor.py
from mimoly_interceptor import ActionEvaluator, Verdict

def test_action_required_when_previous_turn_is_tool():
    evaluator = ActionEvaluator()
    messages = [
        {"role": "user", "content": "Bikin file test.txt"},
        {"role": "assistant", "tool_calls": [{"name": "write_file"}]},
        {"role": "tool", "name": "write_file", "content": "File written"}
    ]
    # Model returns conversational evasion instead of proceeding with tool or final conclusion
    verdict = evaluator.evaluate_turn(messages, tools_present=True, reply_text="Saya akan menulis kodenya nanti ya.", tool_calls=[])
    assert verdict == Verdict.NUDGE_PROSE

def test_conversational_pass_through():
    evaluator = ActionEvaluator()
    messages = [{"role": "user", "content": "Halo, selamat pagi!"}]
    verdict = evaluator.evaluate_turn(messages, tools_present=True, reply_text="Halo! Ada yang bisa saya bantu?", tool_calls=[])
    assert verdict == Verdict.PASS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python projects/mimoly/test_action_interceptor.py`
Expected: FAIL (ModuleNotFoundError: No module named 'mimoly_interceptor')

- [ ] **Step 3: Implement `ActionEvaluator` in `projects/mimoly/mimoly_interceptor.py`**

Implement Option 1 heuristic: check mandatory action triggers (prior tool turn OR technical action keywords) vs pass-through exceptions (greetings, informational queries, completed task conclusions). Support framework-specific directive generation.

- [ ] **Step 4: Run test to verify it passes**

Run: `python projects/mimoly/test_action_interceptor.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/mimoly/mimoly_interceptor.py projects/mimoly/test_action_interceptor.py
git commit -m "feat(interceptor): add strict action evaluator and framework-aware steering"
```

---

### Task 4: Self-Healing Parser & Argument Recovery Updates

**Files:**
- Modify: `projects/mimoly/mimoly.py`
- Test: `projects/mimoly/test_upstream_frames.py`

**Interfaces:**
- Consumes: `smart_extract_tool_calls(reply_text, user_prompt, available_tools)`.
- Produces: Enhanced extraction with A11 (`write_file` dropped content) & A12 (`execute_code` empty `{}`) recovery, and dual-index validation across full client tool catalogue.

- [ ] **Step 1: Write the failing tests in `test_upstream_frames.py`**

Add unit tests for:
1. `write_file` called with missing `content`, where content is harvested from preceding markdown code block.
2. `execute_code` called with `{}` empty dict, where python code is extracted from reasoning/markdown.
3. Invocation of an unselected tool (out of prompt working set) validating successfully against the global client tool list.

- [ ] **Step 2: Run test to verify it fails**

Run: `python projects/mimoly/test_upstream_frames.py`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Update `smart_extract_tool_calls` and `normalize_tool_args` in `projects/mimoly/mimoly.py`**

Enhance the parsing and recovery routines to satisfy all three test cases.

- [ ] **Step 4: Run test to verify it passes**

Run: `python projects/mimoly/test_upstream_frames.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add projects/mimoly/mimoly.py projects/mimoly/test_upstream_frames.py
git commit -m "fix(parser): enhance argument self-healing (A11/A12) and dual-index validation"
```

---

### Task 5: Integration into Main Request Pipeline

**Files:**
- Modify: `projects/mimoly/mimoly.py`
- Test: `projects/mimoly/test_upstream_frames.py`, `projects/mimoly/test_reliability.py`

**Interfaces:**
- Consumes: `ToolRetriever`, `render_adaptive_tool_catalogue`, `ActionEvaluator`.
- Produces: End-to-end integrated `build_agent_prompt`, `sse_generator`, and non-streaming completion flow.

- [ ] **Step 1: Update `build_agent_prompt` in `projects/mimoly/mimoly.py`**

Replace static tier truncation with `ToolRetriever.select_working_set` and `render_adaptive_tool_catalogue`. Store global tools list in request state for parser validation.

- [ ] **Step 2: Wire `ActionEvaluator` into `sse_generator` and non-stream generation**

Check verdict if zero tool calls parsed. If `Verdict.NUDGE_*` and `_content_started is False`:
1. Increment retry counter.
2. Inject steering directive into `upstream_payload["query"]`.
3. Sleep with backoff and retry upstream turn.

- [ ] **Step 3: Run full regression suite**

Run:
```bash
python projects/mimoly/test_reliability.py
python projects/mimoly/test_upstream_frames.py
python projects/mimoly/test_tool_rag.py
python projects/mimoly/test_action_interceptor.py
```
Expected: ALL PASS (exit code 0).

- [ ] **Step 4: Commit**

```bash
git add projects/mimoly/mimoly.py
git commit -m "feat(core): integrate Tool-RAG and Strict Action Interceptor into proxy pipeline"
```

---

### Task 6: E2E Verification & Mock Session Run

**Files:**
- Create: `projects/mimoly/test_rag_agent_e2e.py`

**Interfaces:**
- Runs a multi-turn autonomous coding scenario against local Mimoly server with 250 simulated MCP tools.

- [ ] **Step 1: Write E2E simulation script `projects/mimoly/test_rag_agent_e2e.py`**

Include:
- 250 mock tools (Playwright, Blender, terminal, file, git).
- Turn 1: User asks to build a file -> verify `write_file` called with full content.
- Turn 2: User asks to run tests -> verify `terminal` called.
- Turn 3: User asks for summary -> verify legitimate completion allowed without forced tool call.

- [ ] **Step 2: Execute E2E verification**

Run: `python projects/mimoly/test_rag_agent_e2e.py`
Expected: ALL TURNS PASS with zero dropped arguments and correct finish reasons.

- [ ] **Step 3: Commit**

```bash
git add projects/mimoly/test_rag_agent_e2e.py
git commit -m "test(e2e): add multi-turn autonomous agent verification suite"
```
