# Plan: native toolCalls repair (react/chat)

## Root cause (verified in code)
- `mimoly.py:4104` (stream) / `:4645` (non-stream): `_check_upstream_error_frame`
  runs BEFORE the toolCalls handler (`:4124` / `:4663`).
- `except _UpstreamFrameError: raise` at `:4318-4319` / `:4729-4730` discards
  already-collected toolCalls in `accumulated_chunks` — no retry consult,
  no tool-call probe. Non-stream path returns the 502 seen live.
- Busy-text guards (`:4258-4284` / `:4688-4699`) only cover busy phrases
  inside text, not error frames — the exception bypasses both.
- `should_retry` (mimoly_reliability.py:112-116) knows nothing about
  toolCalls; every retry starts with `accumulated_chunks.clear()`.

## External evidence
- No public project parses native `toolCalls` from `react/chat`
  (all target legacy `bot/chat`). Mimoly is first.
- Trailing empty `{}` matches official server bug XiaomiMiMo/MiMo#103
  (phantom nameless tool call appended after finished answer).
- Official docs: `arguments` is a JSON string; streaming uses index +
  delta append (chat/openai-api, api/chat/responses). No `modelConfig`
  field avoids server-busy.

## Fix (minimal, two points)
1. **Strip trailing `{}` suffix** in `fn_args` at `:4127` / `:4666`
   (or in the parser fallback) before wrapping into
   `<tool_call><function=...>` so `smart_extract_tool_calls` parses it.
2. **Preserve valid toolCalls on post-toolCalls error**: in
   `except _UpstreamFrameError` (`:4318-4319` / `:4729-4730`), probe
   `accumulated_chunks` with `smart_extract_tool_calls` (pattern already
   used at `:4256` / `:4687`); if valid calls exist, continue to normal
   emit (`:4357+` / `:4759+`) instead of raising.

## Tests (TDD, offline, network-deny)
- Replay live shape in stream AND non-stream: single frame with
  `toolCalls: [{name: terminal, arguments: '{"command": ...}{}'}]`
  followed by `event: error` with 服务器繁忙 content.
- Assert: response carries `tool_calls` (terminal, correct args),
  no 502, no leaked error text, single attempt.
- Existing suites must stay green: 62 responsiveness + 5 runners
  (upstream_frames, action_interceptor 15, tool_rag 17, reliability,
  rag_agent_e2e 10) + `hermes verify --json --skip-start`.

## Constraints
- Scoped commit, no broad rewrite, no live network calls.
- Independent review before merge to main.
