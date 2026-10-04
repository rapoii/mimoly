#!/usr/bin/env python
"""End-to-end search + fetch smoke test for mimoly + MCP playwright.

Runs real `hermes chat` sessions against the mimoly provider with the playwright
toolset, then scores each trial from the --format stream-json event stream (not
the model's prose claim).

Two things this harness deliberately accounts for:

* **The model may fetch with `terminal` + curl instead of `browser_navigate`.**
  Both are a real fetch, so a trial PASSES when it executed at least one tool and
  did not fall back to a "no tools available" refusal. Tool names are printed so
  you can see which path it took.
* **A bare search intent still triggers the upstream injector** (the reply starts
  with `webSearch`), so the search prompt names a *search-engine URL* — masking in
  the proxy keeps the injector away and the RSS form gives a citable snapshot.
  A `webSearch`-prefixed answer is counted as FAIL.

Usage:
    python search_fetch_e2e.py            # 3 trials each
    python search_fetch_e2e.py --trials 5

Exit code 0 when every trial in both scenarios did real work.

Prerequisites: mimoly serving on :8080 and
`hermes config set mcp_single_query_discovery_timeout 45` (see README).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.resolve()
HERMES = shutil.which("hermes") or "hermes"

# Fetch: a stable, server-rendered page. The pasted URL is masked by the proxy.
FETCH_PROMPT = (
    "Buka https://en.wikipedia.org/wiki/Xiaomi lalu ringkas isinya. "
    "Sebutkan judul halaman dan 2 fakta dari halaman itu."
)
# Search: name a search-engine URL. Masking keeps the upstream webSearch injector
# away, and Bing's RSS form gives a snapshot the model can actually cite.
SEARCH_PROMPT = (
    "Buka https://www.bing.com/search?q=Xiaomi+MiMo+LLM&format=rss lalu "
    "sebutkan 3 judul hasil beserta URL-nya."
)

# Phrases the model uses when it wrongly believes it has no tools.
_REFUSAL_MARKERS = (
    "tidak tersedia", "tidak punya akses", "no browser tools",
    "tidak benar-benar punya akses", "tidak bisa menjalankan",
)


def run_trial(prompt: str) -> tuple[bool, list[str], str, bool]:
    """Return (did_real_work, tools, final_text, injected)."""
    cmd = [
        HERMES, "chat", "-q", prompt,
        "--provider", "mimoly", "-m", "mimo-v2.6-pro",
        "--reasoning", "none",
        "-t", "terminal,playwright",
        "--format", "stream-json",
        "--ignore-rules", "--max-turns", "40",
    ]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", errors="ignore", timeout=900)

    tools: list[str] = []
    final = ""
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "tool_use":
            tools.append(ev.get("name") or "")
        elif ev.get("type") in ("text", "result") and not final:
            final = str(ev.get("text") or "")

    injected = final.startswith("webSearch")
    refused = any(m in final.lower() for m in _REFUSAL_MARKERS)
    did_work = bool(tools) and not injected and not refused
    return did_work, tools, final, injected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    # Warm the MCP server (diagnostic crutch; the raised bound is the real fix).
    subprocess.run([HERMES, "mcp", "test", "playwright"],
                   capture_output=True, text=True, timeout=300)

    results: dict[str, list[bool]] = {}
    for label, prompt in (("fetch", FETCH_PROMPT), ("search", SEARCH_PROMPT)):
        passes: list[bool] = []
        for i in range(args.trials):
            try:
                ok, tools, final, injected = run_trial(prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"[{label} #{i + 1}] ERROR {exc!r}")
                passes.append(False)
                continue
            passes.append(ok)
            used = ", ".join(tools) if tools else "(none)"
            note = " [webSearch injection]" if injected else ""
            print(f"[{label} #{i + 1}] {'PASS' if ok else 'FAIL'}{note} | tools: {used}")
            if not ok:
                print(f"           final: {final[:180]!r}")
        results[label] = passes

    print("\n=== summary ===")
    exit_code = 0
    for label, passes in results.items():
        n, total = sum(passes), len(passes)
        print(f"{label:7}: {n}/{total} trials did real work")
        if n != total:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
