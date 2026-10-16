# Agent Responsiveness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox syntax for tracking.

**Goal:** Make the general-purpose Mimoly agent reliable and bounded without claiming a model latency SLA.
**Architecture:** Preserve current web gateway. Fix lifecycle and protocol defects with small tested helpers, no broad rewrite. Keep official API capabilities distinct from verified web capabilities.
**Tech Stack:** Existing Python/HTTPX/ASGI stack; standard-library unittest tests (pytest absent in existing venv).
**Spec:** User-approved audit proposal in this document: non-cancelling heartbeat, total request budget, context preservation, conservative interceptor, validated tool parsing, timing and bounded smoke tests.

## Global Constraints
- No credentials in output or source; no edits to session.json/.env or user/global configuration.
- Do not run previous long project/vibe tests or generate demo projects.
- Work on fix/agent-responsiveness-20261016; no push or merge without consent.
- No global installs; use existing .venv/Scripts/python.exe and unittest/script runners.
- Real RED then GREEN evidence for every corrected behavior; bounded offline tests first.
- Never claim proxy changes accelerate upstream inference or imply official API flags work on web endpoints.

## Review Focus
- SSE idle gaps must not cancel pending reads or lose late frames.
- Lock wait, cancellation, busy/nudge backoff and final attempts must obey one request deadline and release resources.
- Retry must retain system/developer/latest goal and recent observations without lossy quote rewriting.
- Natural language answers and explicit no-tool requests must not be forced into tools.
- Parser must reject unknown/example invocations while preserving valid supported protocol examples and pagination.

### Task 1: Lifecycle and protocol reliability hardening
**Files:** Modify mimoly.py, mimoly_reliability.py, mimoly_interceptor.py, mimoly_rag.py as needed; create test_agent_responsiveness.py. Preserve existing tests unless behavior was intentionally corrected, documenting each update.
**Interfaces:** Existing public build_agent_prompt, smart_extract_tool_calls, normalize_tool_args, ActionEvaluator and RetryBudget retain compatibility. Any extracted lifecycle helpers stay small with explicit deadline and cleanup ownership.
- [x] Read affected code and existing runners; record clean baseline for test_upstream_frames.py, test_action_interceptor.py, test_tool_rag.py, test_reliability.py and test_rag_agent_e2e.py, with per-command timeout 60s.
- [x] Add failing unittest regressions for idle SSE then late output; total retry deadline/backoff; cancel while queued and mid-stream; partial-output retry isolation; goal/developer preservation on compact retry; observation quotes and read_file pagination; valid tool_use/input and wrapped OpenAI parsing; unknown/example invocation rejection; conservative interceptor for poetry/arithmetic/no-tool/plan-only and legitimate completion; cache schema fingerprint changes.
- [x] Run regressions and record real expected failures before production edits.
- [x] Implement minimum fixes. One monotonic request deadline starts before preprocessing/lock; use existing normal/ultra budgets unless a documented explicit decision requires change. Wait for pending read without cancelling on heartbeat; cancel/await on exit. Bound all attempts, lock waits and sleeps; no sleep after last attempt; do not retry programming errors or mix attempt outputs. Resource ownership uses try/finally.
- [x] Preserve roles, goal, raw quote content, pagination; compact retries from full instruction/goal context, not messages[-4:]. Reject unknown tool names, support emitted protocol formats, and do not parse illustrative fenced examples as executable calls. Do not force tools based solely on action verbs. Hash complete indexed schema content for retrieval cache identity.
- [x] Add minimal per-request phase/retry timing without prompt or secret logging; distinguish upstream-first-frame, first answer/tool emission and total duration. Document unsupported web output cap rather than pretending max_completion_tokens is enforced.
- [x] Run all five baseline runners plus new unittest regressions. Read modified sections, self-review diff, commit only scoped files. Report exact commands, RED/GREEN output, commit IDs, omissions and risks to docs/superpowers/plans/2026-10-16-agent-responsiveness-report.md.

### Task 2: Independent review and bounded live verification
**Files:** Task 1 diff and report; optional scoped fixes by fresh implementer.
**Interfaces:** Review existing public gateway behavior and deterministic tests, no new feature design.
- [ ] Independent reviewer reads full committed diff and spec, reports compliance AND quality; concurrency/deadline and protocol regressions are blocking.
- [ ] Resolve findings with tests and scoped re-review.
- [ ] Parent verifies final tests. If credentials/service are available, run at most three short live prompts with explicit per-request timeout, no file creation or long agent loop: direct answer, one safe tool action, subsequent tool-result continuation. Compare thinking only if supported by inspected wire payload; no paid endpoint migration.
- [ ] Report measured durations and blockers honestly; do not equate local tests with improved live latency.

## Execution decisions
- User explicitly approved implementation. Use current clean checkout on a new feature branch rather than touching main or creating extra worktrees; no merge/push.
- Implementation is one cohesive task because shared mimoly.py lifecycle and parser edits would conflict across concurrent workers. Independent review follows implementation.
- Baseline pytest is unavailable; use existing script runners and unittest, no dependency installation.
