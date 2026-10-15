"""test_action_interceptor.py - Comprehensive Unit & Concurrency Tests for ActionEvaluator.

Covers:
  1. Mandatory action triggers (in-flight tool turn, Indonesian & English action verbs)
  2. Conversational pass-through (greetings, pleasantries, informational/conceptual queries)
  3. Reasoning-only stall (empty outside <think>, unclosed <think>, whitespace)
  4. Prose evasion (promissory phrases in ID/EN, conversational evasion under action mandate)
  5. Legitimate completion (successful post-tool conclusion without promissory evasion)
  6. Tool calls present & tools not present pass-through
  7. Framework-specific steering directives (hermes, default, claude-code, codex)
  8. Class-level and instance-level method parity
  9. Multi-threaded concurrency safety (zero shared mutable state)
 10. Robustness against malformed or multimodal messages
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Ensure current dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mimoly_interceptor import ActionEvaluator, TurnVerdict, Verdict


def test_action_required_when_previous_turn_is_tool():
    evaluator = ActionEvaluator()
    messages = [
        {"role": "user", "content": "Bikin file test.txt"},
        {"role": "assistant", "tool_calls": [{"name": "write_file"}]},
        {"role": "tool", "name": "write_file", "content": "File written successfully"}
    ]
    # Model returns conversational evasion instead of proceeding with tool or final conclusion
    verdict = evaluator.evaluate_turn(
        messages=messages,
        tools_present=True,
        reply_text="Saya akan menulis kodenya nanti ya.",
        tool_calls=[]
    )
    assert verdict == Verdict.NUDGE_PROSE, f"Expected NUDGE_PROSE, got {verdict}"


def test_conversational_pass_through():
    evaluator = ActionEvaluator()
    messages = [{"role": "user", "content": "Halo, selamat pagi!"}]
    verdict = evaluator.evaluate_turn(
        messages=messages,
        tools_present=True,
        reply_text="Halo! Ada yang bisa saya bantu?",
        tool_calls=[]
    )
    assert verdict == Verdict.PASS, f"Expected PASS for greeting, got {verdict}"

    # Thanks / gratitude
    messages_thanks = [{"role": "user", "content": "Terima kasih banyak atas penjelasannya."}]
    verdict_thanks = evaluator.evaluate_turn(
        messages=messages_thanks,
        tools_present=True,
        reply_text="Sama-sama! Senang bisa membantu.",
        tool_calls=[]
    )
    assert verdict_thanks == Verdict.PASS, f"Expected PASS for gratitude, got {verdict_thanks}"

    # English greeting
    messages_en = [{"role": "user", "content": "Hello! How are you doing today?"}]
    verdict_en = evaluator.evaluate_turn(
        messages=messages_en,
        tools_present=True,
        reply_text="Hello! I am doing well, ready to help you.",
        tool_calls=[]
    )
    assert verdict_en == Verdict.PASS, f"Expected PASS for English greeting, got {verdict_en}"


def test_conceptual_query_pass_through():
    evaluator = ActionEvaluator()

    # Indonesian conceptual queries
    queries_id = [
        "Jelaskan apa itu dependency injection dalam software engineering",
        "Apa perbedaan antara process dan thread di sistem operasi?",
        "Mengapa Python menggunakan GIL?",
        "Bagaimana cara kerja algoritma Dijkstra?",
        "Ceritakan sejarah arsitektur microservices"
    ]
    for q in queries_id:
        msgs = [{"role": "user", "content": q}]
        v = evaluator.evaluate_turn(
            messages=msgs,
            tools_present=True,
            reply_text="Berikut adalah penjelasan mendalam mengenai konsep tersebut...",
            tool_calls=[]
        )
        assert v == Verdict.PASS, f"Expected PASS for conceptual query '{q}', got {v}"

    # English conceptual queries
    queries_en = [
        "Explain how transformer attention mechanism works",
        "What is the difference between TCP and UDP?",
        "Why do we use virtual environments in Python?",
        "How does the BM25 retrieval algorithm calculate term frequencies?",
        "Describe the CAP theorem in distributed systems"
    ]
    for q in queries_en:
        msgs = [{"role": "user", "content": q}]
        v = evaluator.evaluate_turn(
            messages=msgs,
            tools_present=True,
            reply_text="Here is a detailed explanation of the theoretical principles...",
            tool_calls=[]
        )
        assert v == Verdict.PASS, f"Expected PASS for conceptual query '{q}', got {v}"


def test_mandatory_triggers_indonesian_verbs():
    evaluator = ActionEvaluator()
    verbs = [
        "buat", "bikin", "jalankan", "edit", "perbaiki",
        "tulis", "baca", "cari", "eksekusi", "ubah",
        "hapus", "cek", "buka", "test"
    ]
    for verb in verbs:
        prompt = f"Tolong {verb} script index.py sekarang"
        msgs = [{"role": "user", "content": prompt}]
        # Model returns conversational fluff without emitting tool_calls
        v = evaluator.evaluate_turn(
            messages=msgs,
            tools_present=True,
            reply_text="Tentu, saya akan membantu Anda dengan tugas ini. Mari kita pelajari dulu.",
            tool_calls=[]
        )
        assert v == Verdict.NUDGE_PROSE, f"Expected NUDGE_PROSE for Indonesian action verb '{verb}', got {v}"


def test_mandatory_triggers_english_verbs():
    evaluator = ActionEvaluator()
    verbs = [
        "create", "build", "run", "execute", "edit",
        "patch", "fix", "write", "read", "search",
        "modify", "test", "navigate"
    ]
    for verb in verbs:
        prompt = f"Please {verb} the deployment configuration"
        msgs = [{"role": "user", "content": prompt}]
        # Model returns conversational fluff without emitting tool_calls
        v = evaluator.evaluate_turn(
            messages=msgs,
            tools_present=True,
            reply_text="Sure, I can assist you with that. We can start by reviewing the requirements.",
            tool_calls=[]
        )
        assert v == Verdict.NUDGE_PROSE, f"Expected NUDGE_PROSE for English action verb '{verb}', got {v}"


def test_reasoning_only_stall():
    evaluator = ActionEvaluator()
    msgs = [{"role": "user", "content": "Jalankan pytest di direktori tests"}]

    # Case 1: Closed <think> block with empty outside text
    v1 = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text="<think>I should run the terminal tool with pytest command.</think>",
        tool_calls=[]
    )
    assert v1 == Verdict.NUDGE_REASONING, f"Expected NUDGE_REASONING for think-only stall, got {v1}"

    # Case 2: Closed <think> block with whitespace-only outside text
    v2 = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text="<think>Thinking...</think>   \n\n  ",
        tool_calls=[]
    )
    assert v2 == Verdict.NUDGE_REASONING, f"Expected NUDGE_REASONING for whitespace-outside think, got {v2}"

    # Case 3: Completely empty text
    v3 = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text="",
        tool_calls=[]
    )
    assert v3 == Verdict.NUDGE_REASONING, f"Expected NUDGE_REASONING for empty reply, got {v3}"

    # Case 4: Unclosed <think> tag (model cut off or streaming aborted during thinking)
    v4 = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text="<think>Evaluating test suite and checking arguments",
        tool_calls=[]
    )
    assert v4 == Verdict.NUDGE_REASONING, f"Expected NUDGE_REASONING for unclosed think, got {v4}"


def test_prose_evasion_promissory_phrases():
    evaluator = ActionEvaluator()
    # Workflow in flight after tool observation
    msgs = [
        {"role": "user", "content": "Bikin modul auth.py dan testnya"},
        {"role": "assistant", "tool_calls": [{"name": "write_file"}]},
        {"role": "tool", "name": "write_file", "content": "Wrote auth.py"}
    ]

    evasions = [
        "Saya akan menulis testnya sekarang juga.",
        "Berikutnya saya akan menjalankan pengujian.",
        "Nanti saya cek dulu file lognya.",
        "Mari kita jalankan pytest di terminal.",
        "Akan saya eksekusi command berikutnya.",
        "I will now run the test suite to verify.",
        "Let me run pytest to verify the changes.",
        "Next I need to inspect the generated token.",
        "Next step is to patch the config file.",
        "I am going to check if the database is running.",
        "I'm going to read the log file now."
    ]

    for evasion in evasions:
        v = evaluator.evaluate_turn(
            messages=msgs,
            tools_present=True,
            reply_text=evasion,
            tool_calls=[]
        )
        assert v == Verdict.NUDGE_PROSE, f"Expected NUDGE_PROSE for evasion '{evasion}', got {v}"


def test_legitimate_final_completion():
    evaluator = ActionEvaluator()
    msgs = [
        {"role": "user", "content": "Jalankan test suite dan laporkan hasilnya"},
        {"role": "assistant", "tool_calls": [{"name": "terminal"}]},
        {"role": "tool", "name": "terminal", "content": "17 passed in 0.42s"}
    ]

    # Indonesian legitimate conclusion
    reply_id = (
        "Semua 17 unit test telah berhasil dijalankan dengan status PASS (100% sukses). "
        "Tidak ditemukan kegagalan maupun regresi pada modul yang diuji. "
        "Pekerjaan telah selesai dan seluruh artefak telah diverifikasi."
    )
    v_id = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text=reply_id,
        tool_calls=[]
    )
    assert v_id == Verdict.PASS, f"Expected PASS for legitimate Indonesian conclusion, got {v_id}"

    # English legitimate conclusion
    reply_en = (
        "All 17 unit tests passed successfully with 0 failures in 0.42 seconds. "
        "The implementation has been verified and meets all specification criteria. "
        "Here is the final summary of changes applied to the repository."
    )
    v_en = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text=reply_en,
        tool_calls=[]
    )
    assert v_en == Verdict.PASS, f"Expected PASS for legitimate English conclusion, got {v_en}"


def test_tool_calls_present_always_pass():
    evaluator = ActionEvaluator()
    msgs = [{"role": "user", "content": "Bikin file test.txt"}]
    # Even if reply_text has promissory language, having tool_calls means model took action
    v = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=True,
        reply_text="Saya akan membuat filenya sekarang.\n<tool_call><function=write_file></function></tool_call>",
        tool_calls=[{"name": "write_file", "arguments": {"path": "test.txt", "content": "abc"}}]
    )
    assert v == Verdict.PASS, f"Expected PASS when tool_calls is non-empty, got {v}"


def test_tools_not_present_always_pass():
    evaluator = ActionEvaluator()
    msgs = [{"role": "user", "content": "Bikin kode Python untuk kalkulator"}]
    # Client did not provide any tools (tools_present=False)
    v = evaluator.evaluate_turn(
        messages=msgs,
        tools_present=False,
        reply_text="Berikut adalah kode kalkulator dalam Python: ```python\nprint(1+1)\n```",
        tool_calls=[]
    )
    assert v == Verdict.PASS, f"Expected PASS when tools_present=False, got {v}"


def test_framework_specific_steering_directives():
    evaluator = ActionEvaluator()
    tools = ["read_file", "write_file", "terminal", "patch"]

    # Hermes / Default -> Qwen native XML directive
    d_hermes = evaluator.get_steering_directive(framework="hermes", available_tool_names=tools)
    assert "<tool_call>" in d_hermes
    assert "<function=" in d_hermes
    assert "</tool_call>" in d_hermes
    for t in tools:
        assert t in d_hermes

    d_default = ActionEvaluator.get_steering_directive(framework="default", available_tool_names=tools)
    assert "<tool_call>" in d_default

    # Claude Code -> Anthropic JSON tool_use block directive
    d_claude = evaluator.get_steering_directive(framework="claude-code", available_tool_names=tools)
    assert "tool_use" in d_claude
    assert "input" in d_claude
    for t in tools:
        assert t in d_claude

    # Codex -> Strict JSON function calling directive
    d_codex = evaluator.get_steering_directive(framework="codex", available_tool_names=tools)
    assert "tool_calls" in d_codex
    assert "function" in d_codex
    assert "arguments" in d_codex
    for t in tools:
        assert t in d_codex


def test_steering_directive_caps_large_tool_set():
    """Verify tool names in directive are capped to 20 so 250 tools don't bloat prompt."""
    evaluator = ActionEvaluator()
    large_tools = [f"tool_{i}" for i in range(250)]
    for fw in ("hermes", "claude-code", "codex", "default"):
        directive = evaluator.get_steering_directive(framework=fw, available_tool_names=large_tools)
        # First 20 tools should be in the directive
        for i in range(20):
            assert f"tool_{i}" in directive, f"Expected tool_{i} in directive for {fw}"
        # 21st tool and beyond should NOT be in the directive
        assert "tool_20" not in directive, f"tool_20 should be pruned by 20-tool cap for {fw}"
        assert "tool_249" not in directive, f"tool_249 should be pruned by 20-tool cap for {fw}"


def test_class_and_instance_invocation():
    msgs = [{"role": "user", "content": "Bikin file dummy.txt"}]

    # Instance-level
    ev_inst = ActionEvaluator()
    v1 = ev_inst.evaluate_turn(messages=msgs, tools_present=True, reply_text="Saya akan buat nanti.", tool_calls=[])
    d1 = ev_inst.get_steering_directive(framework="hermes")

    # Class-level
    v2 = ActionEvaluator.evaluate_turn(messages=msgs, tools_present=True, reply_text="Saya akan buat nanti.", tool_calls=[])
    d2 = ActionEvaluator.get_steering_directive(framework="hermes")

    assert v1 == v2 == Verdict.NUDGE_PROSE
    assert d1 == d2
    assert TurnVerdict is Verdict


def test_concurrency_thread_safety():
    """Verify zero race conditions or shared mutable state under high concurrency."""
    evaluator = ActionEvaluator()

    test_scenarios = [
        # (messages, reply, tool_calls, expected_verdict)
        ([{"role": "user", "content": "Halo apa kabar"}], "Halo! Baik.", [], Verdict.PASS),
        ([{"role": "user", "content": "Bikin file a.txt"}], "<think>...", [], Verdict.NUDGE_REASONING),
        ([{"role": "user", "content": "Bikin file b.txt"}], "Saya akan bikin nanti", [], Verdict.NUDGE_PROSE),
        ([{"role": "user", "content": "Jalankan test"}], "", [{"name": "terminal"}], Verdict.PASS),
        ([{"role": "user", "content": "Jelaskan OOP"}], "OOP adalah...", [], Verdict.PASS),
        (
            [
                {"role": "user", "content": "Run tests"},
                {"role": "assistant", "tool_calls": [{"name": "terminal"}]},
                {"role": "tool", "content": "OK"}
            ],
            "All tests completed cleanly and successfully.",
            [],
            Verdict.PASS
        ),
    ]

    def _worker(worker_id: int):
        scenario = test_scenarios[worker_id % len(test_scenarios)]
        msgs, reply, t_calls, expected = scenario
        # Call both instance and class methods concurrently
        v_inst = evaluator.evaluate_turn(messages=msgs, tools_present=True, reply_text=reply, tool_calls=t_calls)
        v_cls = ActionEvaluator.evaluate_turn(messages=msgs, tools_present=True, reply_text=reply, tool_calls=t_calls)
        assert v_inst == expected, f"Worker {worker_id} inst mismatch: expected {expected}, got {v_inst}"
        assert v_cls == expected, f"Worker {worker_id} cls mismatch: expected {expected}, got {v_cls}"
        d = ActionEvaluator.get_steering_directive("hermes", ["read_file", "terminal"])
        assert "<tool_call>" in d

    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = [pool.submit(_worker, i) for i in range(200)]
        for f in futures:
            f.result()


def test_robustness_malformed_messages():
    evaluator = ActionEvaluator()

    # Empty messages list
    v_empty = evaluator.evaluate_turn(messages=[], tools_present=True, reply_text="Hello", tool_calls=[])
    assert v_empty == Verdict.PASS

    # Message with None content
    v_none = evaluator.evaluate_turn(
        messages=[{"role": "user", "content": None}],
        tools_present=True,
        reply_text="Hello",
        tool_calls=[]
    )
    assert v_none == Verdict.PASS

    # Multimodal list content
    v_multi = evaluator.evaluate_turn(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Bikin file visual.txt"}]}],
        tools_present=True,
        reply_text="Saya akan kerjakan nanti",
        tool_calls=[]
    )
    assert v_multi == Verdict.NUDGE_PROSE


if __name__ == "__main__":
    test_funcs = [
        test_action_required_when_previous_turn_is_tool,
        test_conversational_pass_through,
        test_conceptual_query_pass_through,
        test_mandatory_triggers_indonesian_verbs,
        test_mandatory_triggers_english_verbs,
        test_reasoning_only_stall,
        test_prose_evasion_promissory_phrases,
        test_legitimate_final_completion,
        test_tool_calls_present_always_pass,
        test_tools_not_present_always_pass,
        test_framework_specific_steering_directives,
        test_steering_directive_caps_large_tool_set,
        test_class_and_instance_invocation,
        test_concurrency_thread_safety,
        test_robustness_malformed_messages,
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
