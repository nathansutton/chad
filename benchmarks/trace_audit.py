"""Audit a saved chad session against the frictions a lean-mode dogfood trace showed.

Reads the newest session JSON under ~/.chad/sessions (or a path given on argv) and
reports the numbers that motivated the lean-arm harness work, so a follow-up run can be
compared to the one before it rather than re-read by hand:

  - system prompt size, split by block (the skills catalog was 82% of it)
  - tool calls by name, and how many landed edits failed to match
  - DEAD round trips: results that are nothing but `[no output]` / `[exit N]`
  - ambient annotations that answered a result in place ([file]/[grep]/[baseline])
  - lever firing counts for the session, read from ~/.chad/session.log

Usage:  .venv/bin/python benchmarks/trace_audit.py [session.json]
"""

import collections
import glob
import json
import os
import re
import sys

SESSIONS = os.path.expanduser("~/.chad/sessions")
SESSION_LOG = os.path.expanduser("~/.chad/session.log")
ANNOTATIONS = ("[file]", "[grep]", "[baseline]", "[session]")


def newest_trace():
    hits = glob.glob(os.path.join(SESSIONS, "*", "*.json"))
    hits = [h for h in hits if not h.endswith("index.json")]
    if not hits:
        sys.exit(f"no session traces under {SESSIONS}")
    return max(hits, key=os.path.getmtime)


def prompt_blocks(system):
    """(chars, heading) per `# Heading` section, largest first."""
    parts = re.split(r"(?m)^(#+ .*)$", system)
    out = [(len(parts[0]), "(preamble)")]
    out += [(len(parts[i + 1]), parts[i].strip()) for i in range(1, len(parts), 2)]
    return sorted(out, reverse=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else newest_trace()
    data = json.load(open(path))
    messages = data["messages"]
    system = messages[0]["content"]

    print(f"trace   {path}")
    print(f"cwd     {data.get('cwd')}   meta {data.get('meta')}\n")

    total = sum(len(m.get("content") or "") for m in messages)
    print(f"system prompt {len(system):>7,} chars   "
          f"({len(system) / total:.0%} of the {total:,}-char final context)")
    for size, head in prompt_blocks(system)[:6]:
        print(f"  {size:>7,}  {head[:70]}")

    calls = collections.Counter()
    for m in messages:
        if m.get("role") == "assistant":
            calls.update(re.findall(r"<function=([a-z_]+)", m.get("content") or ""))
    results = [m for m in messages if m.get("role") == "tool"]
    dead = [m for m in results
            if re.fullmatch(r"\[no output\]|\[exit -?\d+\]\s*", (m.get("content") or "").strip())]
    annotated = [m for m in results
                 if any(a in (m.get("content") or "") for a in ANNOTATIONS)]
    # The literal result prefixes tools.tool_edit returns — matching on a paraphrase
    # silently reported zero failures for a run that had four.
    edit_miss = ("[old string not found", "[old string matches", "[no-op edit")
    edit_fails = [m for m in results
                  if (m.get("content") or "").startswith(edit_miss)]

    print(f"\ntool calls    {sum(calls.values()):>4}   {dict(calls.most_common())}")
    print(f"edit failures {len(edit_fails):>4}   "
          f"({len(edit_fails) / max(calls['edit'], 1):.1%} of {calls['edit']} edits)")
    for m in edit_fails:
        print(f"    {(m.get('content') or '').splitlines()[0][:100]}")
    print(f"DEAD results  {len(dead):>4}   "
          f"({len(dead) / max(len(results), 1):.1%} of {len(results)} results)")
    for m in dead:
        print(f"    {(m.get('content') or '').strip()[:60]!r}")
    print(f"annotated     {len(annotated):>4}   (results the harness answered in place)")
    for m in annotated[:8]:
        for line in (m.get("content") or "").splitlines():
            if line.startswith(ANNOTATIONS):
                print(f"    {line[:110]}")

    if os.path.exists(SESSION_LOG):
        fires = collections.Counter(re.findall(r'"lever": "([a-z_]+)"',
                                               open(SESSION_LOG, errors="replace").read()))
        print("\nlever fires (whole session.log, not just this trace)")
        for name, n in fires.most_common():
            print(f"  {n:>4}  {name}")
    else:
        print(f"\nno {SESSION_LOG} — rerun with CHAD_SESSION_LOG=1 to see lever fires")


if __name__ == "__main__":
    main()
