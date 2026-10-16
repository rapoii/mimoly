# Agent responsiveness — Task 1 implementation report

## Outcome and boundaries

Implemented lifecycle/protocol hardening on `fix/agent-responsiveness-20261016`, based on `d094a17`. **Correction after independent review:** the original preprocessing GREEN was insufficient: its empty-cookie fixture skipped preprocessing and could attempt an upstream client. Prior runs were intended to be offline but were not guaranteed offline; the earlier categorical no-upstream claim is withdrawn. No service restart, dependency installation, global configuration changes, push or merge. No live latency or upstream inference speed was measured; no latency SLA is claimed. The review-fix execution below denies external network and replaces session acquisition with dummy fixtures; no credential contents were inspected during this fix. Task 2 independent re-review/live verification remains parent-owned.

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

## Remaining parser-review corrections (2026-10-16, based on 9b540765)

Both remaining parser blockers have local fixes and behavioral coverage. Independent re-review and parent-owned live verification remain pending; Task 2 is not marked complete. No delegates, live network, service restarts, installs, secrets, push or merge.

### Findings recovery and implementation

Read the ending of `C:/Users/Rafi/AppData/Local/hermes/cache/delegation/live/deleg_905f3cc9/task-0.log`. The log itself contains truncated reviewer text, so the full exact JSON is unavailable even when read directly with Python. Its retained probes show valid write_file content truncated from a fenced-code string to `Before`, and malformed bare XML swallowing following XML/JSON. Reproduced both behaviors with behavioral assertions before their respective production fixes.

- `mimoly.py`: scan candidate starts in order; use `JSONDecoder.raw_decode` to establish complete JSON boundaries before recognizing backticks or protocol tags. JSON strings retain their literal content, including escaped quotes, newlines, triple backticks and protocol-looking text. Wrapper/fence suffixes are consumed only after decoding. Unknown outer JSON does not expose embedded invocations.
- XML legacy fallback now ends malformed bare/wrapped function/invoke spans before the next invocation or fence, retaining the malformed text while allowing subsequent valid calls. Registry/schema checks and sibling-markdown healing remain unchanged.

### Added exact behavioral tests

All are methods of `test_agent_responsiveness.ProtocolTests`:

1. `test_json_string_protocol_markers_are_content`: exact write_file payload and empty-clean-output equality for bare JSON, tool_call-wrapped JSON, standalone JSON fence and wrapped JSON fence; content includes fenced Python, escaped quotes and literal protocol tags.
2. `test_json_string_invocations_are_not_executed`: unknown outer JSON containing a valid-looking nested read_file and fence returns no calls and original text, for bare/wrapped/fenced forms (negative control).
3. `test_malformed_xml_does_not_swallow_next_invocation`: 4 malformed prefixes (unknown function, unknown invoke, known-but-invalid function, wrapped unknown function) × 5 following valid forms (closed function, closed invoke, unclosed function, wrapped JSON, fenced JSON). Assert exactly one correct call and preserve malformed prefix.
4. `test_malformed_xml_alone_is_not_an_invocation`: all 4 malformed prefixes alone produce no call and unchanged original text (negative control).

### Actual RED/GREEN and scoped commit

Existing `.venv/Scripts/python.exe`, each command individually bounded by `timeout 60`:

- JSON RED before production edits: `-m unittest -v test_agent_responsiveness.ProtocolTests.test_json_string_protocol_markers_are_content`; exit 1, `Ran 1 test`, `FAILED (failures=3)`. Evidence: `2026-10-16-parser-json-red.log`. Wrapped-fence positive control already passed.
- XML RED before XML fix: `-m unittest -v test_agent_responsiveness.ProtocolTests.test_malformed_xml_does_not_swallow_next_invocation test_agent_responsiveness.ProtocolTests.test_malformed_xml_alone_is_not_an_invocation`; exit 1, `Ran 2 tests`, `FAILED (failures=18)`. Negative control already passed. Evidence: `2026-10-16-parser-xml-red.log`.
- Final `-m unittest -v test_agent_responsiveness`: exit 0, `Ran 32 tests in 0.559s`, `OK`; evidence `2026-10-16-parser-fixes-green.log`. Existing deny-network guard remains installed before production imports in all six runners.
- Five runners all exit 0: upstream frames / reliability `all checks passed`; interceptor 15 passed/0 failed; tool RAG 17 passed/0 failed; RAG agent E2E 10 passed/0 failed. Full evidence: `2026-10-16-parser-test_<runner>.log` for each runner.
- `-m py_compile mimoly.py test_agent_responsiveness.py` and `git diff --check`: exit 0; existing LF/CRLF checkout warnings only.
- Scoped commit: `fix: preserve JSON payloads and isolate malformed XML calls` (resolve exact hash with `git log -1 --format=%H --grep='fix: preserve JSON payloads and isolate malformed XML calls'`; commit cannot embed its own hash). Changes limited to parser, regressions, this report and eight evidence logs.

No claim of live performance or exhaustive malformed-protocol correctness. Illustrative example detection remains lexical; tolerant legacy XML/JSON repair remains for compatibility. Parent independent re-review/live verification is still outstanding.

## Live-discovered lock ownership correction and offline timeout probes (2026-10-16)

### Live evidence supplied by parent; no new live calls

Parent's two already-completed requests used a temporary server which was subsequently stopped. Direct arithmetic (nonstream, thinking disabled, no tools) returned HTTP 500 at 6.594s with `RuntimeError: Lock is not acquired` in `lifecycle.close -> release_lock -> lock.release`. A safe terminal-tool request (nonstream, thinking disabled) returned HTTP 504 at 50.031s. Supplied timing: attempt start 219ms, upstream first frame 297ms, total 50000ms. Continuation was skipped because no valid terminal invocation returned. First frame is not first answer/tool, and this evidence does NOT establish reasoning latency or identify the live wire payload. No raw upstream frames were retained. This follow-up made no live requests, started no server, installed nothing, and did not inspect credentials.

### Proven cause and scoped fix

`release_account()` combined account reservation/health bookkeeping with unconditional unlocking of any locked account. `_RequestLifecycle` independently owns and tracks acquired locks. A successful response or HTTP early return called `lifecycle.release(account)`, unlocking beneath the lifecycle; the later finally/close attempted the same release and raised. Worse, cancellation/deadline of a request waiting for an account lock released a lock owned by a different request. Checking `locked()` or swallowing RuntimeError is not an ownership fix.

Remove unlocking from `release_account`; only the lifecycle acquisition owner releases locks. Existing stream/nonstream/rotation finally paths already use lifecycle release. Make `close()` explicitly idempotent, including timing emission. No API, parser, retry-policy, or timeout value changed.

The earlier lifecycle fixture mocked both acquire and release, hiding the real integration boundary. New tests use the real `AccountPool`, real acquire/release functions, isolated cache, dummy cookies and scripted upstream, retaining deny-network installation before production imports. Coverage: successful response plus cache hit while a foreign lock is held; HTTP 400 early return; queued timeout and queued cancellation preserving owner lock; reservation release cannot unlock lifecycle ownership; next owner protected after repeated close; close twice emits timing once and releases account once; normal streaming close.

### RED -> GREEN evidence

All commands use existing Python 3.12 `.venv/Scripts/python.exe` and per-command `timeout 60`.

- Before production edits: `-m unittest -v test_agent_responsiveness.RealPoolLifecycleTests`, exit 1, **7 tests; 4 failures and 3 errors**, including the exact live double-release traceback. Full output `2026-10-16-lock-ownership-red.log`.
- After fix and offline probes: `-m unittest -v test_agent_responsiveness`, exit 0, **45 tests, OK** (32 prior + 7 ownership + 6 wire-shape probes). Full output `2026-10-16-lock-ownership-green.log`.
- Five existing runners individually exit 0: `test_upstream_frames.py`, `test_action_interceptor.py`, `test_tool_rag.py`, `test_reliability.py`, `test_rag_agent_e2e.py`. Full output `2026-10-16-lock-test_<runner>.log`.
- `-m py_compile mimoly.py mimoly_reliability.py mimoly_interceptor.py mimoly_rag.py test_agent_responsiveness.py` and `git diff --check`: exit 0. Git's existing LF/CRLF checkout warnings only.

### Deterministic nonstream/no-thinking timeout investigation

Six characterization probes operate through the real request handler with a scripted SSE reader and short 40ms outer deadline (not a 50-second wait). Assertions verify thinking disabled is transmitted to `modelConfig.enableThinking`, invocation validity, number of attempts, and cleanup. These are artificial wire inputs, not reconstructed live responses.

1. Normal `type:text/content` containing a valid tool_call plus `[DONE]` -> valid tool response, no missing parser branch for that shape.
2. Native `toolCalls` with EOF -> valid tool response.
3. Normal text answer with `[DONE]` -> answer 42.
4. Native tool call followed by a stalled read -> 504, one attempt, upstream closed, lock released. Nonstream buffers until EOF/[DONE] before parsing/returning a complete tool. Therefore it can time out even if a usable tool frame has already arrived. This is a read/termination design limitation, not a proven live explanation or asyncio deadlock.
5. `error:{message:...}` without content followed by stall -> 504; error is not surfaced. The normalizer only excludes non-dicts and the reader ignores dictionaries without recognized content/usage/native tool fields. This demonstrates an error-frame gap, not proof the live upstream sent an error.
6. `event: dialogId`, data, blank line, then ordinary text data without another event -> empty answer. `_event_type` is sticky across event boundaries, so subsequent default-event data is skipped. A later explicit `event:` resets it; raw live frames would be needed to establish relevance.

No actual request invokes the declared tools server-side; this gateway returns invocations to the client. The supplied live run never reached the client execution/continuation stage. No tool execution deadlock has been established. Nonstream timing currently records frame arrival, but not answer/tool parsing phases, so absent first-answer timing is not evidence no answer bytes arrived.

### Deadline policy: unresolved, not silently changed

`_request_budget_seconds` currently returns 50s for non-ultraspeed, 200s for ultraspeed, promoted from retry-loop constants into an end-to-end deadline (including preprocessing, lock queueing and reading). Retry scheduling budget is NOT a documented inference SLA. Existing upstream HTTP read timeout is 180s and source comments explicitly allow >60s thinking streams, which conflicts with the new 50s total cap even though this live request had thinking disabled. Fast first-frame timing cannot justify either raising or keeping a universal inference SLA.

Proposed separate policy change for owner decision: introduce a documented positive finite configurable **total request timeout**, independent from retry/backoff budget; define its defaults and whether queue/preprocessing are included; reject zero/NaN/infinity/negative values rather than quietly disabling the bound. Test outer timeout versus retry exhaustion independently for stream and nonstream, including successful response beyond old retry budget, cancellation/cleanup and queued contention. Choose the default from explicit product/client timeout requirements or retained latency evidence, not this one observation. Do not simply increase both retry and inference budgets or globally disable the bound. Parser/error/event fixes should likewise get separate RED->GREEN fixes with verified upstream contracts; the characterization tests deliberately record current gaps rather than disguising speculation as a live root cause.

Remaining: no post-fix live verification by design; safe live retry needs parent/user authorization and should retain sanitized frame **shape/phase** telemetry (not prompt/cookie/response content), and a separate policy decision is required before claiming the 50s behavior resolved.

## Independent-review corrections (2026-10-16)

All three findings fixed in one scoped follow-up commit. No agents spawned, no live requests or server restart. No installs, credential inspection, push or merge.

### Real RED, then GREEN

Before production fixes, installed `offline_test_support.install_network_deny()` before production imports in all six runners. It denies DNS, outbound socket connect/connect_ex/create_connection, and synchronous/asynchronous HTTPX network transports. It raises `BaseException` so production `except Exception` cannot swallow an accidental network attempt. ASGITransport remains available. Windows asyncio requires the stdlib local socketpair/self-pipe; only the synchronous stdlib socketpair construction is exempt, not arbitrary loopback connections. A regression explicitly proves denied connections raise before network use.

- `timeout 60 .venv/Scripts/python.exe -m unittest -v test_agent_responsiveness`: RED exit 1, `Ran 27 tests in 0.451s`, `FAILED (failures=11)` (subtests counted). Output: `2026-10-16-review-fixes-red.log`. Actual failures include preprocessing awaited zero times, syntax-error/sample-file/usage prose suppressing valid calls, mixed JSON/XML losing one invocation, invalid blocks hiding later valid JSON.
- P1 fixture fix: dummy `xiaomichatbot_ph` for preprocessing only; assert preprocessing awaited exactly once, default upstream-client mock never called, response 504, account released. All lifecycle fixtures default to a fail-fast unmocked-client guard. Existing E2E runner now supplies empty dummy session cookies and offline conversation creation instead of reading real session credentials; its client also defaults to fail-fast unless explicitly mocked.
- P1 example-filter fix: only a clear immediately preceding example/usage/syntax introduction ending in a colon excludes a fence. Syntax-error, sample-file and usage-count prose preserve calls; eight explicit introductions remain excluded.
- P2 mixed-protocol fix: extract non-overlapping protocol spans in original order, decode each independently, validate registry/schema, and remove only accepted spans. Outer wrappers own nested content to avoid duplicate extraction. Keep legacy XML/JSON repair and sibling-markdown argument healing (first runner pass caught loss of write_file/execute_code healing; fixed before final verification).
- Additional targeted RED: unclosed invalid tool_call could hide later valid JSON, `2026-10-16-review-unclosed-red.log`, exit 1, `Ran 1 test`, one failure. Stop malformed wrapper spans at the next protocol boundary.
- Additional targeted RED: wrapped JSON fence cleanup left a dangling closing tag, `2026-10-16-review-wrapped-red.log`, exit 1, `Ran 1 test`, one failure. Recognize the complete outer wrapper as one span.

### Final verification

Every command used existing `.venv/Scripts/python.exe`, individual `timeout 60`, no pytest installation. Final results:

| Command arguments | Actual output | Exit |
|---|---|---|
| `-m unittest -v test_agent_responsiveness` | 28 tests, OK | 0 |
| `test_upstream_frames.py` | all checks passed | 0 |
| `test_action_interceptor.py` | 15 passed, 0 failed | 0 |
| `test_tool_rag.py` | 17 passed, 0 failed | 0 |
| `test_reliability.py` | all checks passed | 0 |
| `test_rag_agent_e2e.py` | 10 passed, 0 failed | 0 |

Final regression log: `2026-10-16-review-fixes-green.log`; existing-runner logs: `2026-10-16-review-test_<runner>.log`. Tool-return elapsed measurements for the batch were 1.674s, 2.782s, 3.466s, 4.181s, 5.078s, 6.591s respectively (includes tool scheduling overhead; not application latency). Compile and whitespace checks were run before commit. No current test proves live latency or upstream behavior. Lexical example recognition remains deliberately bounded, not full intent understanding. Fresh independent re-review remains parent-owned (user prohibited spawning agents here).
