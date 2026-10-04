#!/usr/bin/env python
"""End-to-end search + fetch smoke test for mimoly + MCP playwright.

Runs real `hermes chat` sessions against the mimoly provider with the
playwright toolset, then counts how many trials actually EXECUTED a browser
tool (evidence from the --format stream-json event stream, not the model's
prose claim).

Usage:
    python search_fetch_e2e.py            # 3 trials each
    python search_fetch_e2e.py --trials 5

Exit code 0 when every trial in both scenarios executed a tool.

Known upstream caveat (see web2api skill, "auto webSearch injection"):
Xiaomi MiMo Studio sometimes answers a search-intent prompt from an injected
web-retrieval payload instead of calling the tool, and the model then claims
it has no browser tools. That is an upstream behaviour, not a mimoly bug, so
this script reports a pass RATE rather than asserting 100%.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parent.resolve()
HERMES = shutil.which("hermes") or "hermes"

# A fetch prompt names a stable, server-rendered page and never asks the model
# to "search", so the upstream webSearch injector stays out of the way.
FETCH_PROMPT = (
    "Buka https://en.wikipedia.org/wiki/Xiaomi memakai tool browser "
    "browser_navigate, lalu panggil browser_snapshot, lalu sebutkan judul "
    "halaman dan 2 fakta dari halaman itu."
)
# A search prompt deliberately contains no URL: pasting a URL makes the
# upstream inject its own (usually empty) retrieval payload, which derails the
# model. Let the model choose the engine.
SEARCH_PROMPT = (
    "Cari di web informasi terbaru tentang Xiaomi MiMo LLM memakai tool "
    "browser. Panggil browser_navigate ke mesin pencari pilihanmu, lalu "
    "browser_snapshot, lalu sebutkan 3 judul hasil beserta URL-nya."
)


def run_trial(prompt: str) -> tuple[bool, list[str], str]:
    """Run one chat session; return (executed_any_browser_tool, tools, final_text)."""
    cmd = [
        HERMES, "chat", "-q", prompt,
        "--provider", "mimoly", "-m", "mimo-v2.6-pro",
        "--reasoning", "none",
        "-t", "terminal,playwright",
        "--format", "stream-json",
        "--ignore-rules", "--max-turns", "40",
    ]
    env = dict(os.environ)
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True,
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
            name = ev.get("name") or ""
            tools.append(name)
        elif ev.get("type") in ("text", "result") and not final:
            final = str(ev.get("text") or "")
    browser = [t for t in tools if "browser" in t or "playwright" in t]
    return bool(browser), tools, final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    # Warm the MCP server so the first trial does not race cold-start discovery.
    subprocess.run([HERMES, "mcp", "test", "playwright"],
                   capture_output=True, text=True, timeout=300)

    results: dict[str, list[bool]] = {}
    for label, prompt in (("fetch", FETCH_PROMPT), ("search", SEARCH_PROMPT)):
        passes: list[bool] = []
        for i in range(args.trials):
            try:
                ok, tools, final = run_trial(prompt)
            except Exception as exc:  # noqa: BLE001
                print(f"[{label} #{i + 1}] ERROR {exc!r}")
                passes.append(False)
                continue
            passes.append(ok)
            used = ", ".join(tools) if tools else "(none)"
            print(f"[{label} #{i + 1}] {'PASS' if ok else 'FAIL'} | tools called: {used}")
            if not ok:
                print(f"           final: {final[:180]!r}")
        results[label] = passes

    print("\n=== summary ===")
    exit_code = 0
    for label, passes in results.items():
        n, total = sum(passes), len(passes)
        print(f"{label:7}: {n}/{total} trials executed a browser tool")
        if n != total:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
