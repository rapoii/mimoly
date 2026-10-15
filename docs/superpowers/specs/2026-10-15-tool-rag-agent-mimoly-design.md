# Design Document: Mimoly 100% AI Agent Architecture with In-Memory Tool-RAG & Strict Action Interceptor

- **Date:** 2026-10-15
- **Author:** Lily & Rafi Permana
- **Status:** Draft / Under Review
- **Target Repository:** `projects/mimoly`

---

## 1. Executive Summary & Intent

Mimoly is a high-performance Web2API proxy that bridges Xiaomi's MiMo web LLM cluster into OpenAI- and Anthropic-compatible endpoints for autonomous agent harnesses (specifically Hermes Agent, Claude Code, and Codex).

Currently, when rich agent environments expose 200+ tools (e.g., standard file/terminal tools plus MCP Playwright, Blend-AI, Context7, and Cronjob), Mimoly must aggressively compress the tool definitions to fit MiMo's 20,000–26,000 character prompt budget. Under this pressure, Mimoly collapses tool definitions into bare names without parameter schemas (Tier 3). Consequently, the model hallucinates parameter names, drops arguments (`write_file` without `content`, `execute_code` with empty `{}`), or lapses into conversational prose instead of executing actions.

This design introduces a **Two-Tier Working Set Tool-RAG** architecture combined with a **Strict Action Interceptor** inside Mimoly. By decoupling tool discovery from execution and enforcing actionable output, Mimoly guarantees zero-miss tool calling and reliable autonomous execution, while remaining completely stateless, sub-millisecond fast, and capable of scaling to 1000+ concurrent users without heavy external vector infrastructure.

---

## 2. Goals, Non-Goals, and Anti-Goals

### 2.1 Goals
1. **Zero-Miss Tool Calling (100% Agent Reliability):** Eliminate missed tool calls, dropped arguments (A11/A12), and reasoning-only stalls (`<think>` dead ends).
2. **In-Memory Working Set Tool-RAG:** Dynamically curate tools per turn into a lean working set (~8–12 tools):
   - **Anchor Tools (5 tools):** Core primitives (`read_file`, `write_file`, `patch`, `terminal`, `search_files`) always included.
   - **Recent Memory (1–3 tools):** Tools executed in the previous turn are automatically retained to prevent task disruption.
   - **Dynamic RAG (3–5 tools):** High-relevance MCP tools retrieved via in-memory BM25/keyword semantic scoring matching the user's latest prompt and previous tool output.
3. **Adaptive Full Schema Injection:** Always supply the complete JSON schema (properties, types, descriptions, required fields) and a minimal few-shot invocation example for every active working set tool.
4. **Strict Action Interceptor:** Detect when a task requires technical action (e.g., modifying files, executing code, tool follow-up) but upstream MiMo returns conversational prose or stops in `<think>`. The proxy intercepts the response and triggers an immediate, bounded auto-nudge retry requiring an actionable tool invocation.
5. **High-Concurrency Stateless Scaling (1000+ Users):** Ensure zero per-user cross-contamination, sub-2ms retrieval overhead, no disk/vector DB dependencies, and safe resource utilization.
6. **Robust Fault Tolerance & Healing:** Maintain backward compatibility with existing account pools, circuit breakers, rate limiters, and response normalization.

### 2.2 Non-Goals
1. **External Vector Database:** No Milvus, ChromaDB, Qdrant, or heavy embedding models. The indexing must run purely in memory inside the Python process per request or cached across static tool definitions.
2. **File System / Project Document RAG:** Mimoly does not index user codebases or workspace documents; that responsibility remains with the autonomous agent (Hermes) via `search_files` and `read_file`.
3. **Modifying Upstream Client Protocols:** No breaking changes to standard `/v1/chat/completions` or Anthropic `/v1/messages` schemas.

### 2.3 Anti-Goals
1. **Over-Forcing Conversational Turns:** Never force a tool call if the user is merely greeting, asking an informational/conceptual question, or when the agent has legitimately completed all work and is delivering a final report.
2. **Tool Thrashing / Infinite Loops:** Avoid repeatedly retrying or nudging the model when an upstream failure is structural or when a tool repeatedly produces identical errors.

---

## 3. System Architecture & Detailed Components

```
+-----------------------------------------------------------------------------------+
|                                 Hermes Agent / Client                             |
+-----------------------------------------------------------------------------------+
                                          |
                        HTTP POST /v1/chat/completions
                        (Tools: 200+ schemas, Messages: dialogue)
                                          v
+-----------------------------------------------------------------------------------+
|                                MIMOLY PROXY ENGINE                                |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | 1. Dynamic Tool Registry & Working Set Retriever (BM25 + Semantic Match)   |  |
|  |    - Anchor Tools: [terminal, read_file, write_file, patch, search_files]   |  |
|  |    - Active Working Set: Anchor + Last-Used Tools + Top-K Relevant Tools    |  |
|  +-----------------------------------------------------------------------------+  |
|                                         |                                         |
|  +-----------------------------------------------------------------------------+  |
|  | 2. Adaptive Prompt Assembler & Full Schema Injector                         |  |
|  |    - Injects FULL JSON Schema for Active Working Set                        |  |
|  |    - Generates Qwen XML / JSON Few-Shot Guidance                            |  |
|  |    - Renders Budget-Safe Action Ledger & Dialogue History                   |  |
|  +-----------------------------------------------------------------------------+  |
|                                         |                                         |
|  +-----------------------------------------------------------------------------+  |
|  | 3. Upstream Dispatcher & Account Pool (Xiaomi FastChat / Ultraspeed)        |  |
|  +-----------------------------------------------------------------------------+  |
|                                         |                                         |
|  +-----------------------------------------------------------------------------+  |
|  | 4. Response Interceptor & Strict Action Evaluator                           |  |
|  |    - Case A: Valid Tool Call -> Format to OpenAI/Anthropic chunk -> Stream  |  |
|  |    - Case B: Legitimate Final Answer -> Stream to Client                   |  |
|  |    - Case C: Action Needed but Prose/Reasoning-Stall emitted               |  |
|  |      => Intercept -> Trigger Auto-Nudge Retry (max 2 attempts)              |  |
|  +-----------------------------------------------------------------------------+  |
|                                         |                                         |
|  +-----------------------------------------------------------------------------+  |
|  | 5. Output Parser & Argument Self-Healer                                     |  |
|  |    - Coerce types, repair dropped args (A11/A12), strip XML artifacts     |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                          |
                              Streamed SSE Response
                                          v
+-----------------------------------------------------------------------------------+
|                                 Hermes Agent / Client                             |
+-----------------------------------------------------------------------------------+
```

---

### 3.1 Component 1: Working Set Tool Retriever (`ToolRetriever`)

#### Working Set Composition
For any request with $N$ total tools ($N$ can exceed 250):
1. **Anchor Tier (Mandatory):**
   - Fixed identifiers: `{"terminal", "read_file", "write_file", "patch", "search_files", "execute_code", "clarify"}`.
   - Any of these present in the incoming payload are unconditionally preserved in the working set.
2. **Contextual History Tier (Retention):**
   - The retriever scans the last 3 assistant turns in `messages`. Any tool called within these turns is unconditionally preserved to maintain operational continuity.
3. **Dynamic Discovery Tier (RAG):**
   - The remaining tools (e.g. MCP Playwright, Blend-AI, Context7) are tokenized and scored using an in-memory BM25 / token-overlap engine.
   - The search query is derived from:
     $$Q = \text{last\_user\_message} + \text{" "} + \text{last\_tool\_observation\_preview}$$
   - Top $K$ ($K \in [3, 6]$ based on remaining prompt budget) tools are retrieved.

#### In-Memory BM25 Indexing
- Built on the fly per request or cached per unique hash of the tool list.
- Stop-words filtered; tokens stem-matched or lowercased.
- Overhead: $< 1.5$ ms for 300 tool definitions on Python 3.12/3.14.

#### Dual-Index Validation Policy (Builder's Call)
- The prompt only renders the active working set ($\le 12$ tools) to protect MiMo from prompt bloat.
- However, `smart_extract_tool_calls` and `normalize_tool_args` validate against the **complete caller tool registry** ($N$ tools). If the model invokes a valid tool outside the current turn's working set (e.g. from instructions or earlier context), it is accepted and validated against its full schema without rejection.

---

### 3.2 Component 2: Adaptive Schema & Few-Shot Injector

Because the active working set contains only $\le 12$ tools instead of 200+, the prompt overhead drops from $>15,000$ characters to $<3,500$ characters.
This allows Mimoly to provide **Full Parameter Schemas**:
- Exact parameter names.
- Type definitions (string, integer, array of objects, etc.).
- Parameter descriptions and `required` arrays.
- Compact single-line XML/JSON invocation examples for tricky tools (e.g., `execute_code`, `patch`, `write_file`).

#### Schema Overflow Pruning Policy (Builder's Call)
If a working set of 12 tools contains verbose MCP schemas exceeding the 4,000-character tool budget:
1. First, truncate parameter descriptions to 80 characters while preserving property names, types, and required lists.
2. If still over budget, dynamically decrease $K$ (e.g. from 5 dynamic tools to 3) until the total schema fits cleanly under the budget ceiling.

---

### 3.3 Component 3: Strict Action Interceptor

#### Activation Heuristic (Rafi Permana's Decision — Option 1: Contextual & Safe)
The Strict Action Interceptor evaluates whether an action is mandatory before allowing a text-only turn:
1. **Mandatory Action Triggers:**
   - The immediate previous turn in dialogue was a `tool` observation (workflow in-flight).
   - OR the user prompt contains concrete technical action verbs/directives:
     - Indonesian: `buat`, `bikin`, `jalankan`, `edit`, `perbaiki`, `tulis`, `baca`, `cari`, `eksekusi`, `ubah`, `hapus`, `cek`, `buka`, `test`.
     - English: `create`, `build`, `run`, `execute`, `edit`, `patch`, `fix`, `write`, `read`, `search`, `modify`, `test`, `navigate`.
2. **Pass-Through Exceptions (No Interception):**
   - Pure conversational greetings (`halo`, `hi`, `terima kasih`).
   - Conceptual / informational questions without action imperative (`jelaskan konsep X`, `apa itu Y`).
   - Final report completion: the previous turn was a successful tool run AND the current turn provides a comprehensive conclusion without promissory evasion phrases (*"I will now...", "Let me run...", "Saya akan..."*).

#### Streaming Delivery Policy (Builder's Call)
- In streaming mode, if `_content_started` is False (only reasoning tokens `<think>` have been emitted, or no chunks emitted at all), an auto-nudge retry is completely safe.
- Mimoly discards the aborted turn internally and streams the second attempt cleanly.
- If answer text already started streaming to the client, retry is locked out to avoid corrupting client SSE state machines.

#### Framework-Aware Directive Alignment (Builder's Call)
The auto-nudge prompt dynamically adapts to the client's `framework`:
- **Default / Hermes:** Qwen native XML directive (`<tool_call><function=...`).
- **Claude Code:** Anthropic JSON tool_use block directive.
- **Codex:** Strict JSON function calling directive in English.

#### Decision Matrix for Auto-Nudge
When upstream MiMo responds, the interceptor categorizes the turn:
1. **Success (Tool Emitted):** Parser successfully extracts 1 or more tool calls $\rightarrow$ stream immediately.
2. **Success (Legitimate Final Completion):**
   - Preceding turn was a tool execution that satisfied the task, AND
   - Model provides a complete summary/explanation without phrases like "I will now read...", "Let me run...", or "Next I need to...".
   - $\rightarrow$ stream text directly.
3. **Defect (Reasoning-Only / `<think>` Stall):**
   - Upstream output is empty outside `<think>...</think>`.
   - $\rightarrow$ Trigger Auto-Nudge.
4. **Defect (Prose Evasion under Action Mandate):**
   - Dialogue indicates an active multi-step technical command, but model returns conversational fluff (e.g., *"Sure, I can help with that. To do this, we should write the file..."*) without issuing `<tool_call>`.
   - $\rightarrow$ Intercept chunk emission, append forceful steering directive.
   - Retry with account rotation if rate-limited. Max retries: 2.

---

### 3.4 Component 4: Self-Healing Parser Enhancements

1. **A11 Repair (`write_file` dropped content):**
   - If `write_file` has `path` but missing `content`, harvest content from preceding markdown code blocks or unassigned properties.
2. **A12 Repair (`execute_code` empty `{}`):**
   - If `execute_code` has empty `code` or `{}` arguments, extract the latest Python code block from the reasoning/prose text and inject it into `code`.
3. **Raw XML / JSON Drift:**
   - Handle unescaped quotes inside XML parameters gracefully using regex boundary matching rather than naive `json.loads`.

---

## 4. Concurrency, Performance & Scalability (1000+ Users)

1. **Zero Shared Mutable State:**
   - The `ToolRetriever` operates strictly as a pure function or per-request instance.
   - No disk writes, no global locks during retrieval.
2. **Fast In-Memory Operation:**
   - BM25 index generation for 250 tools takes $\approx 1.2$ ms.
   - Query ranking takes $\approx 0.3$ ms.
   - Total latency added to request pipeline: $< 2.0$ ms.
3. **Memory Footprint:**
   - Each tool index structure is $< 100$ KB.
   - Garbage collected immediately after prompt assembly. 1,000 concurrent requests add $< 100$ MB of transient RAM.

---

## 5. Verification & Testing Strategy

1. **Unit Tests (`test_tool_rag.py`):**
   - Validate Anchor Tool preservation (never dropped).
   - Validate Recent Tool retention across turns.
   - Validate BM25 query precision against synthetic MCP tools (Playwright, Blender, DB, Shell).
   - Benchmark ranking latency under 500 candidate tools.
2. **Integration Tests (`test_action_interceptor.py`):**
   - Mock upstream emitting reasoning-only stall $\rightarrow$ verify intercept & retry payload.
   - Mock upstream emitting prose evasion $\rightarrow$ verify directive injection.
   - Verify legitimate completion is NOT blocked.
3. **E2E Agent Session (`test_mimoly_e2e.py`):**
   - Multi-turn run matching real Hermes behavior with 200+ mock MCP tools.
   - Zero dropped arguments, zero invalid tool calls.

---

## 6. Implementation Deliverables
1. `mimoly_rag.py`: Pure Python in-memory BM25 retriever & working set manager.
2. Integration into `mimoly.py`:
   - Replace static truncation in `build_agent_prompt` with `ToolRetriever`.
   - Wire `Strict Action Interceptor` into both stream and non-stream generation loops.
   - Integrate updated argument coercion into `smart_extract_tool_calls`.
3. Full test suites confirming 100% pass rate before production deployment.
