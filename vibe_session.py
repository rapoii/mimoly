#!/usr/bin/env python
"""Drive a realistic multi-turn `hermes chat` session (vibe-coder style).

Each turn is a separate `hermes chat -q ... --resume <sid>` invocation, exactly
like a human coming back to the terminal. All stream-json is saved per turn so
we can later count which tools / MCP servers were actually used.

Usage:
    python vibe_session.py <turns.json> <outdir> [--tools a,b,c] [--model X]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERMES = "hermes"


def run_turn(prompt, tools, model, provider, resume, timeout, cwd, image=None, max_turns=30):
    cmd = [
        HERMES, "chat", "-q", prompt,
        "--provider", provider, "-m", model,
        "--reasoning", "none",
        "-t", tools,
        "--format", "stream-json",
        "--ignore-rules", "--max-turns", str(max_turns),
    ]
    if image:
        cmd += ["--image", image]
    if resume:
        cmd += ["--resume", resume]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="ignore", timeout=timeout)
    return proc.stdout or "", proc.stderr or "", time.time() - t0


def parse(out):
    sid, tools, text = None, [], ""
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            sid = ev.get("session_id")
        if ev.get("type") == "tool_use":
            tools.append(ev.get("name") or "")
        if ev.get("type") in ("text", "result") and not text:
            text = str(ev.get("text") or "")
    return sid, tools, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("turns_file")
    ap.add_argument("outdir")
    ap.add_argument("--tools", default="terminal,file")
    ap.add_argument("--model", default="mimo-v2.6-pro")
    ap.add_argument("--provider", default="mimoly")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--cwd", default=".")
    args = ap.parse_args()

    turns = json.loads(Path(args.turns_file).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sid = None
    summary = []
    for i, turn in enumerate(turns, 1):
        label = turn.get("label", f"turn{i}")
        prompt = turn["prompt"]
        extra_tools = turn.get("tools")
        tools = extra_tools or args.tools
        extra_img = turn.get("image")
        print(f"\n########## TURN {i} [{label}] ##########", flush=True)
        print(f"PROMPT: {prompt[:300]}", flush=True)
        p = prompt
        try:
            out, err, dur = run_turn(p, tools, args.model, args.provider, sid,
                                     args.timeout, args.cwd, image=extra_img,
                                     max_turns=args.max_turns)
        except subprocess.TimeoutExpired:
            print(f"[{label}] TIMEOUT after {args.timeout}s", flush=True)
            summary.append({"turn": i, "label": label, "timeout": True})
            continue
        (outdir / f"turn{i:02d}_{label}.jsonl").write_text(out, encoding="utf-8")
        if err.strip():
            (outdir / f"turn{i:02d}_{label}.stderr").write_text(err, encoding="utf-8")
        got_sid, tlist, text = parse(out)
        if got_sid:
            sid = got_sid
        unknown = out.count("Unknown toolset")
        print(f"[{label}] dur={dur:.0f}s sid={sid} tools={tlist} unknown_toolset={unknown}",
              flush=True)
        print(f"[{label}] text: {text[:300]}", flush=True)
        summary.append({"turn": i, "label": label, "sid": sid, "tools": tlist,
                        "unknown_toolset": unknown, "dur": round(dur),
                        "text": text[:500]})
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
