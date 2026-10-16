# Agent responsiveness — Task 1 implementation report

## Outcome and boundaries

Implemented offline lifecycle/protocol hardening on `fix/agent-responsiveness-20261016`, based on `d094a17`. No upstream HTTP calls, service restart, dependency installation, global configuration changes, push or merge. Session/.env contents were not read or output. No live latency or upstream inference speed was measured; no latency SLA is claimed. Task 2 independent review/live verification remains parent-owned.

## Baseline (before production edits)

Existing `.venv/Scripts/python.exe` reports Python 3.12.13. Pytest is absent; used script runners and standard-library unittest.

```bash
for f in test_upstream_frames.py test_action_interceptor.py test_tool_rag.py test_reliability.py test_rag_agent_e2e.py; do
  timeout 60 .venv/Scripts/python.exe "$f"
done
```

Initial run: all five exited successfully; upstream/reliability reported all checks passed, interceptor 15 passed/0 failed, tool RAG 17 passed/0 failed, RAG agent E2E 10 passed/0 failed. These are bounded offline scripted tests, not the former long project/vibe workload.

## RED evidence

Created `test_agent_responsiveness.py` before production edits. Initial real execution:

```bash
timeout 60 .venv/Scripts/python.exe -m unittest -v test_agent_responsiveness
```

Initial 12-test run exited 1 with failed assertions (including subtests); final reproducible baseline RED counts are recorded below. Failures showed lost late SSE frame, queued-cancellation account leak, stale partial response mixed into successful retry, retry of ValueError/programming defect, developer context omission, quote rewriting, pagination removal, unsupported emitted protocols, unknown/example execution, action-verb overreach, stale retrieval cache and excessive retry delay.

Additional deadline/compact-retry regressions were added while implementing the shared lifecycle. To verify them against actual original code without reverting the checkout, extracted only the four production modules with `git show d094a17:<filename>` into Hermes scratch and put that directory first on `sys.path`; reused the current test module from the repository. No dependency or secret files were copied. Command:

```bash
timeout 60 .venv/Scripts/python.exe C:/Users/Rafi/AppData/Local/hermes/cache/scratch/mimoly_baseline_regressions.py
```

Persisted output: `2026-10-16-responsiveness-red.log`: `Ran 22 tests in 0.984s`, `FAILED (failures=26, errors=4)`. The four errors are expected outer test timeouts: baseline ignores the injected request-wide budget while waiting for lock, preprocessing, busy/nudge sleep or pending stream read. Current implementation answers with bounded 504/SSE timeout. Some preservation/control tests already passed baseline (midstream cancellation, streamed-answer guard, multiple JSON calls); they remain controls, not claimed original defects.

Self-review also produced real targeted RED before subsequent fixes:
- `... -m unittest -v test_agent_responsiveness.ProtocolTests.test_multiple_json_calls_preserved`: failed `['a.py'] != ['a.py', 'b.py']`; fixed collecting all JSON blocks rather than returning first.
- `... -m unittest -v test_agent_responsiveness.LifecycleTests.test_explicit_stream_close_releases_resources`: failed lock still held; fixed explicit inner iterator closure at admission wrapper.
- `... -m unittest -v test_agent_responsiveness.LifecycleTests.test_streamed_reasoning_is_not_mixed_with_retry`: second answer appeared after already-visible reasoning; fixed guard on any visible reasoning/answer/tool output.
- `... -m unittest -v test_agent_responsiveness.ProtocolTests.test_instruction_roles_remain_distinguishable`: missing `[System]` marker; labeled system/developer roles in flattened web prompt.

An initial heartbeat fixture lacked an ASGI receive channel; fixed the test request to provide a cancellation-aware receive coroutine. This was a test-fixture issue, not a production defect.

## Modified sections

- `mimoly.py`: small request lifecycle owns monotonic deadline/account reservations/owned locks, beginning before request preprocessing and lock wait. Existing 50s standard / 200s ultraspeed budgets retained. Outer bounded awaits cover preprocessing, lock/throttle/conversation/HTTP, all retry sleeps and stream reads. Retry budgets inherit the same start instant. Owned resources are released on cancellation, timeout, normal return and generator close; rotation releases previous lock before replacement.
- SSE read helper keeps exactly one pending `__anext__` across heartbeat intervals (`asyncio.wait`, not cancelling `wait_for` per heartbeat), cancels/awaits on helper exit; `contextlib.aclosing` owns it on loop break. Admission wrapper explicitly closes inner SSE iterator.
- Attempts clear buffered response before retry. Retry only retryable transport failures, never arbitrary programming errors. Once reasoning/answer/tool output is visible, retry cannot concatenate a different attempt. Final-attempt busy/nudge paths cannot sleep then fall off empty.
- Compact retry rebuilds from full message instruction/goal context instead of `messages[-4:]`, with tighter existing ultraspeed prompt budget. System/developer instructions remain distinguishable. Tool observation double quotes and `read_file` pagination survive.
- Parser validates against declared registry/schema, rejects unknown names, supports tool_use/input and wrapped OpenAI functions (bare, XML wrapped or standalone JSON fence), preserves multiple JSON invocations and legacy Qwen XML/code recovery. Illustrative fenced examples are removed from executable parsing, while original natural language is retained when no calls accepted.
- `mimoly_interceptor.py`: explicit no-tool/plan-only intent wins; action verbs alone cannot mandate tools. Only promissory evasion triggers prose nudge; substantive completion/poem/arithmetic can pass. Added existing English "I can assist" evasion wording to preserve positive regression.
- `mimoly_rag.py`: cache key SHA-256 over complete canonical serialized indexed tools, not `(name, description length)`.
- `mimoly_reliability.py`: backoff capped to remaining budget, zero after final attempt. Existing public signatures retained.
- Timing output contains phase/retry durations only: upstream first frame, first answer/tool emission, total and attempt starts; no prompt/session/response logging introduced. Reasoning is not counted as first answer. Removed misleading unused default `max_completion_tokens=8192`: inspected web wire payload has no verified output-token-cap field; client max_tokens/max_completion_tokens are not enforced by this web gateway.
- `test_upstream_frames.py`: intentionally updated old quote-mutation assertion to exact raw observation preservation; declared read_file in a fixture formerly testing an undeclared name; changed obsolete source-string assertion `_stream_started` to visible-output guard. All other existing runners unchanged.
- `test_agent_responsiveness.py`: short mocked HTTP/request lifecycle tests and protocol/unit regressions.

## GREEN and static verification

```bash
timeout 60 .venv/Scripts/python.exe -m unittest -v test_agent_responsiveness
for f in test_upstream_frames.py test_action_interceptor.py test_tool_rag.py test_reliability.py test_rag_agent_e2e.py; do
  timeout 60 .venv/Scripts/python.exe "$f"
done
.venv/Scripts/python.exe -m py_compile mimoly.py mimoly_reliability.py mimoly_interceptor.py mimoly_rag.py test_agent_responsiveness.py
git diff --check
```

New regression run: `Ran 22 tests in 0.552s`, `OK`, exit 0; full output in `2026-10-16-responsiveness-green.log`. Final five runner output is persisted in `2026-10-16-responsiveness-runners.log` (each command individually bounded to 60s). Compile and diff whitespace checks exit 0. Git emits existing checkout LF/CRLF conversion warnings only.

## Risks / honest limits for independent review

- No live browser/web upstream test; real late-frame, credential rotation and load behavior still need bounded parent verification. Local timing is not evidence of faster upstream inference.
- The web endpoint is still a flattened prompt, not native role/structured tool API. Role markers improve retention, not a guarantee of upstream instruction hierarchy.
- Existing finite prompt/history budgets still truncate very large instructions/observations and retain recent rather than entire history. Quote loss is fixed; this is not unlimited lossless context storage.
- Illustrative-fence detection is conservative lexical screening, not a complete natural-language intent parser. Standalone protocol fences remain executable for compatibility; independent review should inspect ambiguous mixed prose/protocol cases.
- Cancellation is cooperative: synchronous CPU-bound preprocessing cannot be preempted mid-Python call. The monotonic budget expires before further awaited work; cleanup time may extend beyond deadline. No wall-clock SLA is claimed.
- Existing broad error rendering, account health semantics and preexisting logging remain; newly introduced timing logs contain durations only.
- A StreamingResponse returned but never iterated/closed by its ASGI owner cannot run generator cleanup; normal ASGI consumption, cancellation and explicit post-start close are tested.

## Commits

- Starting baseline: `d094a17`.
- Implementation/tests: `282143f` — `fix: bound agent lifecycle and validate protocol retries`.
- Report, captured RED/GREEN logs, and Task 1 checkbox completion: subsequent scoped `docs: record responsiveness verification evidence` commit (resolve with `git log -1 --format=%H -- docs/superpowers/plans/2026-10-16-agent-responsiveness-report.md`; its own hash cannot be embedded without changing it).
- No push/merge. Task 2 remains unchecked and parent-owned.
