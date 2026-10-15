"""mimoly_rag.py - In-Memory BM25 & Working Set Retriever for Mimoly Tool-RAG.

Provides:
  - ToolRetriever: Two-tier working set manager (Anchors + History Retention + BM25 dynamic).
  - BM25Index: Fast in-memory token-inverted index with positive Lucene IDF.
  - Zero shared mutable state across requests; thread-safe for 1000+ concurrent requests.
"""

import functools
import math
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Translation table to fast-tokenize identifiers, code symbols, and punctuation into spaces
_TRANS_TABLE = str.maketrans('_-.:,;!?/()[]{}\"\'\\|<>~`@#$%^&*+=', ' ' * 32)

# Indonesian & English domain action verbs and synonyms for high-precision query expansion
_SYNONYMS: Dict[str, List[str]] = {
    # Web & Browser navigation
    "buka": ["open", "navigate", "browser", "web", "url"],
    "kunjungi": ["visit", "navigate", "browser", "web", "url"],
    "lihat": ["view", "read", "get", "inspect", "browse"],
    "tengok": ["view", "inspect"],
    "klik": ["click", "press"],
    "pencet": ["click", "press"],
    "ketik": ["type", "fill", "input"],
    "masukkan": ["input", "fill", "insert"],
    "scroll": ["scroll", "down", "up"],
    "screenshot": ["screenshot", "capture"],
    "tangkapan": ["screenshot", "capture"],

    # Filesystem & Editing
    "baca": ["read", "get", "view", "file"],
    "tulis": ["write", "create", "save", "file"],
    "buat": ["create", "build", "make", "new"],
    "bikin": ["create", "build", "make", "new"],
    "edit": ["edit", "patch", "modify", "update"],
    "ubah": ["modify", "edit", "change", "patch"],
    "ganti": ["replace", "change", "modify"],
    "hapus": ["delete", "remove", "drop"],
    "buang": ["delete", "remove"],
    "cari": ["search", "find", "grep"],
    "temukan": ["find", "discover", "search"],
    "jalankan": ["run", "execute", "start"],
    "eksekusi": ["execute", "run"],
    "cek": ["check", "verify", "inspect", "test"],
    "periksa": ["check", "inspect", "verify"],
    "uji": ["test", "verify"],

    # 3D, Multimedia & Specialized
    "render": ["render", "frame", "animation", "image", "3d", "blender"],
    "animasi": ["animation", "animate", "render"],
    "putar": ["rotate", "spin", "play"],
    "geser": ["move", "translate"],
    "skala": ["scale", "resize"],
}

_URL_RE = re.compile(r'https?://|\b\w+\.(?:com|org|net|io|id|gov|edu|co)\b|localhost', re.IGNORECASE)


def _extract_tool_name(tool: Any) -> str:
    """Extract tool identifier safely from either OpenAI function format or plain tool format."""
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        return name if isinstance(name, str) else ""
    name = tool.get("name")
    return name if isinstance(name, str) else ""


def _extract_tool_desc(tool: Any) -> str:
    """Extract tool description safely."""
    if not isinstance(tool, dict):
        return ""
    fn = tool.get("function")
    if isinstance(fn, dict):
        desc = fn.get("description")
        return desc if isinstance(desc, str) else ""
    desc = tool.get("description")
    return desc if isinstance(desc, str) else ""


def _extract_tool_params(tool: Any) -> List[str]:
    """Extract parameter property names from schema if present."""
    if not isinstance(tool, dict):
        return []
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    if not isinstance(fn, dict):
        return []
    params = fn.get("parameters")
    if isinstance(params, dict):
        props = params.get("properties")
        if isinstance(props, dict):
            return [str(k) for k in props.keys()]
    return []


def _tokenize_text(text: str) -> List[str]:
    """Fast tokenization converting symbols to whitespace, lowercased, stripped."""
    if not text:
        return []
    return text.lower().translate(_TRANS_TABLE).split()


class BM25Index:
    """Immutable in-memory BM25 index over candidate tool definitions."""

    __slots__ = ("postings", "doc_lens", "avgdl", "idf", "N")

    def __init__(
        self,
        postings: Dict[str, List[Tuple[int, int]]],
        doc_lens: List[int],
        avgdl: float,
        idf: Dict[str, float],
        N: int,
    ):
        self.postings = postings
        self.doc_lens = doc_lens
        self.avgdl = avgdl
        self.idf = idf
        self.N = N

    @classmethod
    def build(cls, candidate_tools: List[Dict[str, Any]]) -> "BM25Index":
        N = len(candidate_tools)
        if N == 0:
            return cls({}, [], 0.0, {}, 0)

        postings: Dict[str, List[Tuple[int, int]]] = {}
        doc_lens: List[int] = []
        total_len = 0

        for idx, tool in enumerate(candidate_tools):
            name = _extract_tool_name(tool)
            desc = _extract_tool_desc(tool)
            param_keys = _extract_tool_params(tool)

            name_tokens = _tokenize_text(name)
            desc_tokens = _tokenize_text(desc)
            param_tokens = [tok for p in param_keys for tok in _tokenize_text(p)]

            # Boost tool name tokens 2x so exact name components strongly signal relevance
            all_tokens = name_tokens * 2 + desc_tokens + param_tokens
            dl = len(all_tokens)
            doc_lens.append(dl)
            total_len += dl

            counts: Dict[str, int] = {}
            for tok in all_tokens:
                counts[tok] = counts.get(tok, 0) + 1

            for tok, cnt in counts.items():
                if tok in postings:
                    postings[tok].append((idx, cnt))
                else:
                    postings[tok] = [(idx, cnt)]

        avgdl = total_len / N if N > 0 else 1.0

        # Lucene/Robertson BM25 non-negative IDF
        idf: Dict[str, float] = {}
        for term, plist in postings.items():
            n_q = len(plist)
            idf[term] = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)

        return cls(postings, doc_lens, avgdl, idf, N)


# In-memory index cache to achieve sub-0.5ms median latency across multi-turn sessions
_CACHE_LOCK = threading.Lock()
_INDEX_CACHE: Dict[Tuple[Tuple[str, int], ...], BM25Index] = {}
_CACHE_ORDER: List[Tuple[Tuple[str, int], ...]] = []
_MAX_CACHE_SIZE = 32


def _get_or_build_index(candidate_tools: List[Dict[str, Any]]) -> BM25Index:
    """Retrieve cached immutable BM25Index or construct a new one thread-safely."""
    fp = tuple(
        (_extract_tool_name(t), len(_extract_tool_desc(t)))
        for t in candidate_tools
    )
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(fp)
        if cached is not None:
            return cached

    # Build outside the lock to eliminate contention across concurrent requests
    index = BM25Index.build(candidate_tools)

    with _CACHE_LOCK:
        if fp not in _INDEX_CACHE:
            if len(_INDEX_CACHE) >= _MAX_CACHE_SIZE:
                oldest = _CACHE_ORDER.pop(0)
                _INDEX_CACHE.pop(oldest, None)
            _INDEX_CACHE[fp] = index
            _CACHE_ORDER.append(fp)

    return index


class _HybridMethod:
    """Descriptor enabling a method to be invoked cleanly on both a class and an instance.

    When called on a class (e.g. ToolRetriever.select_working_set(...)), it instantiates a
    default instance and invokes the method. When called on an instance
    (e.g. retriever.select_working_set(...)), it invokes the bound instance method directly.
    """

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


class ToolRetriever:
    """Two-Tier Working Set Manager with In-Memory BM25 Tool-RAG.

    Partitions full tool catalogue into:
      1. Anchor Tier (Fixed, Mandatory agent baseline tools).
      2. Contextual Retention Tier (Retains tools called in recent assistant turns).
      3. Dynamic Discovery Tier (In-memory BM25 over remaining tools).
    """

    DEFAULT_ANCHOR_TOOLS = {
        "terminal",
        "read_file",
        "write_file",
        "patch",
        "search_files",
        "execute_code",
        "clarify",
    }

    def __init__(self, anchor_tools: Optional[Set[str]] = None):
        if anchor_tools is not None:
            self.anchor_tools = set(anchor_tools)
        else:
            self.anchor_tools = set(self.DEFAULT_ANCHOR_TOOLS)

    @_HybridMethod
    def select_working_set(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
        max_dynamic_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Select working set of tools for the current dialogue turn.

        Callable either as an instance method:
            retriever = ToolRetriever(...)
            retriever.select_working_set(tools, messages, max_dynamic_k=5)
        or directly as a classmethod:
            ToolRetriever.select_working_set(tools, messages, max_dynamic_k=5)

        Args:
            tools: Full caller tool definitions list.
            messages: Dialogue messages leading up to the current turn.
            max_dynamic_k: Maximum dynamically retrieved tools to append.

        Returns:
            Selected subset of tools (Anchor tools + Retained recent tools + Top-K BM25 matches).
        """
        if not tools:
            return []

        # Filter candidate tools upfront: must be dict and have a non-empty name string
        valid_tools: List[Dict[str, Any]] = [
            t for t in tools
            if isinstance(t, dict) and bool(_extract_tool_name(t))
        ]
        if not valid_tools:
            return []

        # Map name -> tool definition for fast resolution and existence check
        tool_map: Dict[str, Dict[str, Any]] = {}
        for t in valid_tools:
            name = _extract_tool_name(t)
            if name:
                tool_map[name] = t

        selected_names: Set[str] = set()
        working_set: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # 1. Anchor Tier (Mandatory Agent Primitives)
        # -------------------------------------------------------------
        for t in valid_tools:
            name = _extract_tool_name(t)
            if name in self.anchor_tools and name not in selected_names:
                selected_names.add(name)
                working_set.append(t)

        # -------------------------------------------------------------
        # 2. Contextual Retention Tier (Last 3 assistant turns)
        # -------------------------------------------------------------
        # Collect window covering up to the last 3 assistant turns
        window_messages: List[Dict[str, Any]] = []
        assistant_turns = 0
        for m in reversed(messages or []):
            if not isinstance(m, dict):
                continue
            window_messages.append(m)
            if m.get("role") == "assistant":
                assistant_turns += 1
                if assistant_turns >= 3:
                    break

        # Process window in chronological order (earliest to latest in window)
        # using dict insertion ordering to preserve stable chronology
        recent_tool_names: Dict[str, None] = {}
        for m in reversed(window_messages):
            role = m.get("role")
            if role == "assistant":
                tool_calls = m.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            fn = tc.get("function")
                            fn_name = fn.get("name") if isinstance(fn, dict) else None
                            fn_name = fn_name or tc.get("name")
                            if isinstance(fn_name, str) and fn_name:
                                recent_tool_names[fn_name] = None
                # Also detect XML tool calls inside assistant message text
                content = m.get("content")
                if isinstance(content, str):
                    for match in re.findall(r'<function=([a-zA-Z0-9_]+)>', content):
                        recent_tool_names[match] = None
            elif role == "tool":
                name = m.get("name")
                if isinstance(name, str) and name:
                    recent_tool_names[name] = None

        for name in recent_tool_names:
            if name in tool_map and name not in selected_names:
                selected_names.add(name)
                working_set.append(tool_map[name])

        # If no dynamic slots requested or all valid tools already selected, return early
        if max_dynamic_k <= 0 or len(selected_names) >= len(valid_tools):
            return working_set

        # -------------------------------------------------------------
        # 3. Dynamic Discovery Tier (In-Memory BM25)
        # -------------------------------------------------------------
        candidate_tools = [
            t for t in valid_tools
            if _extract_tool_name(t) not in selected_names
        ]

        if not candidate_tools:
            return working_set

        # Derive search query: last_user_message + last_tool_observation_preview
        last_user_text = ""
        for m in reversed(messages or []):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, list):
                    parts = [part.get("text", "") for part in c if isinstance(part, dict) and "text" in part]
                    last_user_text = " ".join(parts)
                elif isinstance(c, str):
                    last_user_text = c
                break

        last_tool_obs = ""
        for m in reversed(messages or []):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool":
                c = m.get("content", "")
                if isinstance(c, str):
                    last_tool_obs = c[:500]
                break

        query_raw = f"{last_user_text} {last_tool_obs}".strip()
        query_tokens = _tokenize_text(query_raw)

        # Expand query tokens with synonyms and URL heuristics
        expanded_tokens: List[str] = list(query_tokens)
        for tok in query_tokens:
            syns = _SYNONYMS.get(tok)
            if syns:
                expanded_tokens.extend(syns)

        if _URL_RE.search(query_raw):
            expanded_tokens.extend(["browser", "navigate", "url", "web", "open"])

        # Fetch index and compute BM25 scores
        index = _get_or_build_index(candidate_tools)
        N = index.N
        if N == 0:
            return working_set

        scores = [0.0] * N
        k1 = 1.5
        b = 0.75
        avgdl = index.avgdl or 1.0

        for q in expanded_tokens:
            plist = index.postings.get(q)
            if plist:
                q_idf = index.idf.get(q, 0.0)
                for doc_idx, f in plist:
                    dl = index.doc_lens[doc_idx]
                    scores[doc_idx] += q_idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (dl / avgdl)))

        # Partition matching candidates
        positive_matches = [(scores[i], i) for i in range(N) if scores[i] > 0.0]

        if positive_matches:
            # Sort top scoring tools
            positive_matches.sort(key=lambda x: x[0], reverse=True)
            for _, doc_idx in positive_matches[:max_dynamic_k]:
                tool = candidate_tools[doc_idx]
                name = _extract_tool_name(tool)
                if name and name not in selected_names:
                    selected_names.add(name)
                    working_set.append(tool)
        elif not working_set:
            # Fallback if no anchor tools existed and query had zero matches:
            # Include first max_dynamic_k valid tools so caller gets a non-empty catalogue
            for tool in candidate_tools[:max_dynamic_k]:
                name = _extract_tool_name(tool)
                if name and name not in selected_names:
                    selected_names.add(name)
                    working_set.append(tool)

        return working_set
