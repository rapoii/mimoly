#!/usr/bin/env python
"""Summarise which tools / MCP servers a vibe_session run actually used.

Reads every *.jsonl in the outdir, counts tool_use events by name, groups MCP
tool names (mcp__<server>__<tool>) by server, and prints a per-turn + overall
table. Also reports how many turns hit the cold-start race (Unknown toolset).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

outdir = sys.argv[1]
files = sorted(glob.glob(os.path.join(outdir, "turn*.jsonl")))

overall = Counter()
per_server = Counter()
turns = []
race_hits = 0

for f in files:
    name = os.path.basename(f)
    label = re.sub(r"^turn\d+_", "", name).replace(".jsonl", "")
    tools = []
    unknown = 0
    sid = None
    for line in open(f, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line.startswith("{"):
            if "Unknown toolset" in line:
                unknown += 1
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            sid = ev.get("session_id")
        if ev.get("type") == "tool_use":
            tools.append(ev.get("name") or "")
    for t in tools:
        overall[t] += 1
        m = re.match(r"mcp__([a-zA-Z0-9_]+)__", t)
        if m:
            per_server[m.group(1)] += 1
    if unknown:
        race_hits += 1
    turns.append((label, sid, tools, unknown))

print("=== per turn ===")
for label, sid, tools, unknown in turns:
    uniq = sorted(set(tools))
    print(f"{label:20} | tools={len(tools):2} uniq={len(uniq):2} "
          f"race={'YES' if unknown else 'no'} | {', '.join(uniq) if uniq else '(none)'}")

print("\n=== overall tool usage (count) ===")
for t, c in overall.most_common():
    print(f"  {c:3}  {t}")

print("\n=== MCP servers used ===")
for s, c in per_server.most_common():
    print(f"  {s:12} {c} calls")
if not per_server:
    print("  (none)")

print(f"\n=== distinct tools used: {len(overall)} / turns: {len(turns)} "
      f"/ turns hit by race: {race_hits} ===")
