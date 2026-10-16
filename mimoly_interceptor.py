"""mimoly_interceptor.py - Strict Action Interceptor & Framework-Aware Steering Directive Engine.

Provides:
  - Verdict / TurnVerdict (enum): PASS, NUDGE_REASONING, NUDGE_PROSE.
  - ActionEvaluator: Pure in-memory evaluator implementing Option 1 heuristic.
    - Evaluates in-flight tool workflows and user technical action directives.
    - Accurately passes pure conversational greetings and conceptual queries.
    - Distinguishes reasoning stalls (<think> dead ends) from prose evasion.
    - Approves legitimate post-tool completion summaries.
    - Generates framework-aware steering directives (Hermes XML, Claude Code, Codex).
  - Thread-safe and stateless; designed for 1000+ concurrent requests.
"""

import enum
import functools
import re
from typing import Any, Callable, Dict, List, Optional, Set


class Verdict(str, enum.Enum):
    """Evaluation verdict for an assistant turn."""
    PASS = "PASS"
    NUDGE_REASONING = "NUDGE_REASONING"
    NUDGE_PROSE = "NUDGE_PROSE"


TurnVerdict = Verdict


class _HybridMethod:
    """Descriptor enabling a method to be invoked cleanly on both a class and an instance."""

    def __init__(self, func: Callable[..., Any]):
        self.func = func
        functools.update_wrapper(self, func)

    def __get__(self, instance: Any, owner: Optional[type] = None) -> Callable[..., Any]:
        if instance is None:
            @functools.wraps(self.func)
            def _caller(*args: Any, **kwargs: Any) -> Any:
                if args and isinstance(args[0], owner):  # type: ignore
                    return self.func(args[0], *args[1:], **kwargs)
                return self.func(owner(), *args, **kwargs)  # type: ignore

            return _caller

        @functools.wraps(self.func)
        def _bound_caller(*args: Any, **kwargs: Any) -> Any:
            return self.func(instance, *args, **kwargs)

        return _bound_caller


# Regex for Indonesian technical action verbs & common inflections (formal and informal vibe-coder suffixes like -in)
_ID_ACTION_VERBS = re.compile(
    r'\b(buat|buatkan|buatin|membuat|'
    r'bikin|bikinkan|bikinin|membikin|'
    r'jalan|jalankan|jalanin|menjalankan|'
    r'nyalakan|nyalain|menyalakan|'
    r'pasang|pasangkan|pasangin|memasang|'
    r'edit|ngedit|mengedit|'
    r'perbaiki|benerin|memperbaiki|'
    r'tulis|tuliskan|tulisin|menulis|'
    r'baca|bacakan|bacain|membaca|'
    r'cari|carikan|cariin|mencari|'
    r'hitung|hitungkan|hitungin|menghitung|'
    r'eksekusi|mengeksekusi|'
    r'ubah|mengubah|'
    r'hapus|hapuskan|hapusin|menghapus|'
    r'cek|cekin|mengecek|'
    r'periksa|periksakan|periksain|memeriksa|'
    r'buka|bukakan|bukain|membuka|'
    r'simpan|simpankan|simpen|simpenin|menyimpan|'
    r'tanya|tanyakan|tanyain|menanyakan|'
    r'setup|set up|setting|'
    r'bagi|bagikan|bagiin|membagi|'
    r'test|mengetes|uji|menguji)\b',
    re.IGNORECASE
)

# Regex for English technical action verbs & common inflections
_EN_ACTION_VERBS = re.compile(
    r'\b(create|creates|creating|created|'
    r'build|builds|building|built|'
    r'run|runs|running|ran|'
    r'execute|executes|executing|executed|'
    r'edit|edits|editing|edited|'
    r'patch|patches|patching|patched|'
    r'fix|fixes|fixing|fixed|'
    r'write|writes|writing|wrote|'
    r'read|reads|reading|'
    r'search|searches|searching|searched|'
    r'modify|modifies|modifying|modified|'
    r'test|tests|testing|tested|'
    r'navigate|navigates|navigating|navigated)\b',
    re.IGNORECASE
)

# Regex for conceptual/informational query indicators
_CONCEPTUAL_STARTERS = re.compile(
    r'^\s*(jelaskan|apa itu|apa beda|apa perbedaan|mengapa|kenapa|bagaimana cara kerja|bagaimana|'
    r'apa yang dimaksud|ceritakan|uraikan|terangkan|'
    r'explain|what is|what are|difference between|why do|why does|why is|why are|why|'
    r'how does|how do|how to|how is|how are|describe|tell me about)\b',
    re.IGNORECASE
)

# Imperative conjunctions that link explanation to mandatory action execution
_IMPERATIVE_CONJUNCTIONS = re.compile(
    r'\b(lalu|kemudian|setelah itu|dan|then|and then|after that)\s+'
    r'(perbaiki|bikin|buat|jalankan|tulis|eksekusi|ubah|hapus|fix|create|build|run|edit|patch|write|execute)\b',
    re.IGNORECASE
)

# Promissory evasion phrases (model promises action in prose instead of calling a tool)
_PROMISSORY_EVASIONS = re.compile(
    r'\b('
    r'((saya|gue|gw|aku|kami|kita)\s+(akan|mau|bakal|langsung|hendak|bikin|bikinin|buat|buatin|siapin|kerjain|jalanin|eksekusi|patch|tulis|baca|cek|periksa|cari))\b|'
    r'berikutnya saya akan|selanjutnya saya akan|nanti saya|mari kita jalankan|'
    r'akan saya|sekarang saya akan|langkah selanjutnya|kita akan|saya mau|saya hendak|'
    r'i will now|i will next|let me run|let me check|let me execute|let me read|'
    r'let me write|let me test|let me edit|let me patch|let me inspect|next i need to|'
    r'next i will|next step is to|i am going to|i\'m going to|now i will|i will proceed|'
    r'i will create|i will write|i will execute|i will read|i will run|i will modify|'
    r'i will inspect|i will test|i will check|i can assist|we will now|let\'s run|let\'s check|'
    r'let\'s execute|let\'s inspect)\b',
    re.IGNORECASE
)


def _strip_thinking(text: str) -> str:
    """Strips <think>...</think> reasoning blocks, including unclosed tags."""
    if not text:
        return ""
    # Strip closed think blocks
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip unclosed think block (if streaming was interrupted or model cut off mid-thought)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _extract_text(content: Any) -> str:
    """Safely extracts plain string from string, multimodal list, or dict representations."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(str(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    if content is None:
        return ""
    return str(content)


class ActionEvaluator:
    """Strict Action Interceptor and Evaluator.

    Implements Option 1 heuristic to enforce mandatory tool actions when appropriate,
    while safely passing pure conversational turns, conceptual queries, and legitimate
    completed task reports.
    """

    def __init__(self) -> None:
        pass

    @_HybridMethod
    def evaluate_turn(
        self,
        messages: List[Dict[str, Any]],
        tools_present: bool = True,
        reply_text: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        framework: str = "default",
    ) -> Verdict:
        """Evaluates whether an assistant turn is compliant or requires an auto-nudge intercept.

        Args:
            messages: Dialogue history up to current turn.
            tools_present: True if client provided tools in request payload.
            reply_text: Raw output emitted by upstream model.
            tool_calls: Parsed tool calls emitted by model (if any).
            framework: Agent harness framework (hermes, claude-code, codex, default).

        Returns:
            Verdict: PASS, NUDGE_REASONING, or NUDGE_PROSE.
        """
        # 1. If client provided no tools, model cannot call tools -> PASS
        if not tools_present:
            return Verdict.PASS

        # 2. If model emitted at least one valid tool call -> PASS
        if tool_calls and len(tool_calls) > 0:
            return Verdict.PASS

        # 3. Analyze dialogue context
        cleaned_reply = _strip_thinking(reply_text)

        # Check if immediate previous turn is a tool observation (in-flight workflow)
        immediate_previous_turn_is_tool = False
        if messages:
            for msg in reversed(messages):
                role = msg.get("role")
                if role in ("system",):
                    continue
                if role in ("tool", "function"):
                    immediate_previous_turn_is_tool = True
                break

        # Extract last user message
        user_text = ""
        if messages:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_text = _extract_text(msg.get("content", ""))
                    break

        # Check for technical action verbs in user prompt
        has_id_action = bool(_ID_ACTION_VERBS.search(user_text))
        has_en_action = bool(_EN_ACTION_VERBS.search(user_text))
        has_action_verb = has_id_action or has_en_action

        # Check if user prompt is a conceptual / informational query without action imperative
        is_conceptual = False
        if user_text and _CONCEPTUAL_STARTERS.search(user_text):
            if not _IMPERATIVE_CONJUNCTIONS.search(user_text):
                # Prompt starts with conceptual indicator and lacks imperative action commands
                is_conceptual = True

        # Explicit text-only and planning requests override inferred workflow intent.
        if re.search(r"do not (?:use tools|execute)|no tools|without tools|plan only|plan.*only|tanpa tool|jangan (?:gunakan tool|eksekusi)", user_text, re.I):
            return Verdict.PASS

        # Case A: In-flight tool workflow (immediate previous turn was tool execution)
        if immediate_previous_turn_is_tool:
            # If reply has no text outside <think>, model stalled in reasoning
            if not cleaned_reply:
                return Verdict.NUDGE_REASONING

            # If model uses promissory evasion phrases instead of acting or concluding
            if _PROMISSORY_EVASIONS.search(cleaned_reply):
                return Verdict.NUDGE_PROSE

            # Legitimate final completion: substantive conclusion without promissory evasion
            return Verdict.PASS

        # Case B: Conceptual or informational queries without action imperative -> PASS
        if is_conceptual:
            return Verdict.PASS

        # Case C: User prompt requested concrete technical action
        if has_action_verb:
            # Empty text outside <think> is a reasoning stall
            if not cleaned_reply:
                return Verdict.NUDGE_REASONING

            # A verb alone does not prove tools are necessary (poetry/math).
            # Only intercept an explicit promise to act, not a substantive answer.
            if _PROMISSORY_EVASIONS.search(cleaned_reply):
                return Verdict.NUDGE_PROSE
            return Verdict.PASS

        # Case D: Non-action conversational turns (greetings, pleasantries, questions) -> PASS
        return Verdict.PASS

    @_HybridMethod
    def get_steering_directive(
        self,
        framework: str = "default",
        available_tool_names: Optional[List[str]] = None,
    ) -> str:
        """Generates a framework-specific steering directive for auto-nudging the model.

        Args:
            framework: Target agent framework ('hermes', 'claude-code', 'codex', 'default').
            available_tool_names: Optional list of active working set tool names.

        Returns:
            str: Forceful directive requiring immediate tool execution.
        """
        fw = (framework or "default").lower().strip()
        tool_hint = ""
        if available_tool_names:
            tool_hint = f"\nAvailable tools: {', '.join(available_tool_names[:20])}"

        if fw == "claude-code":
            return (
                "[ACTION REQUIRED: EXECUTE TOOL NOW]\n"
                "Do not output conversational text or evasion. Emit an Anthropic tool_use block immediately:\n"
                "```json\n"
                '{"type": "tool_use", "id": "call_1", "name": "tool_name", "input": {"param_name": "value"}}\n'
                "```\n"
                f"Do not talk about taking action. Execute the tool directly now.{tool_hint}"
            )

        if fw == "codex":
            return (
                "[ACTION REQUIRED: EXECUTE TOOL NOW]\n"
                "Do not output conversational text or promissory evasion. Emit an OpenAI tool_calls JSON block immediately:\n"
                "```json\n"
                '{"tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "tool_name", "arguments": "{\\"param_name\\": \\"value\\"}"}}]}\n'
                "```\n"
                f"Do not talk about taking action. Return the function call directly now.{tool_hint}"
            )

        # Hermes / Default (Qwen native XML format)
        return (
            "[ACTION REQUIRED: EXECUTE TOOL NOW]\n"
            "Do not explain, converse, or promise what you will do. Execute the necessary tool call immediately using XML:\n"
            "<tool_call>\n"
            "<function=tool_name>\n"
            "<parameter=param_name>value</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            f"Do not talk about taking action. Output the <tool_call> block directly now.{tool_hint}"
        )
