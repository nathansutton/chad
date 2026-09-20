#!/usr/bin/env python3
"""How much of chad's wall clock is search? The denominator for any search lever.

A faster grep is worth what grep costs, and chad already times every tool call it makes
into ~/.chad/session.log. This reads that log rather than a benchmark, so the answer is
what the agent actually did, not what a harness asked it to do.

Two things make the naive number wrong, and both are corrected here:

  * A command is counted as a search only when EVERY pipeline segment is a read-only
    text tool. `rg -n x src/ && npm test` is a test run wearing a grep's clothes; charging
    its wall clock to search inflates the answer by three orders of magnitude.
  * The denominator is model decode, recovered from the per-step `N tok @ R tok/s` lines,
    plus tool execution. Tool time alone flatters any tool-side lever.

It also reports the corpus-size distribution, because an index only pays for itself if it
is reused, and a directory chad visits once cannot amortize one.

Run:
    uv run python benchmarks/tgrep/workload.py
"""

import glob
import json
import os
import re
import subprocess

TOOL_LINE = re.compile(r"INFO TOOL (\w+)\((\{.*?\})\)\s*->.*?\[(\d+\.\d+)s\]\s*$")
STEP_LINE = re.compile(r"INFO step \d+: (\d+) tok @ ([\d.]+) tok/s")
SEGMENT = re.compile(r"\|\||&&|[|;&\n]")
LOGS = os.path.expanduser("~/.chad/session.log*")
SESSIONS = os.path.expanduser("~/.chad/sessions/*/*.json")

# Read-only text tools. A command built only from these did nothing but look.
READ_ONLY = frozenset((
    "rg", "grep", "egrep", "fgrep", "ag", "ack", "fd", "find", "locate", "sed", "awk",
    "head", "tail", "cut", "sort", "uniq", "wc", "cat", "ls", "tr", "xargs", "echo",
    "true", "nl", "basename", "dirname", "realpath", "printf", "column", "tee", "comm",
    "diff", "rev", "date",
))
SEARCHERS = frozenset(("rg", "grep", "egrep", "fgrep", "ag", "ack"))


def head_word(segment: str) -> str:
    """The program a pipeline segment runs, past sudo/env-assignment noise."""
    segment = re.sub(r"^\s*\(*\s*", "", segment)
    segment = re.sub(r"^\s*(?:sudo|command|time|env)\s+", "", segment)
    segment = re.sub(r"^\s*\w+=\S+\s+", "", segment)
    match = re.match(r"([\w./-]+)", segment)
    return os.path.basename(match.group(1)) if match else ""


def wall_clock() -> None:
    decode_s = 0.0
    steps = 0
    bash_n = 0
    bash_s = 0.0
    tool_s = 0.0
    search_n = 0
    search_s = 0.0
    compound_n = 0
    compound_s = 0.0
    latencies = []
    for path in sorted(glob.glob(LOGS)):
        with open(path, errors="replace") as fh:
            for line in fh:
                step = STEP_LINE.search(line)
                if step:
                    rate = float(step.group(2))
                    steps += 1
                    decode_s += int(step.group(1)) / rate if rate else 0.0
                    continue
                tool = TOOL_LINE.search(line.rstrip())
                if not tool:
                    continue
                secs = float(tool.group(3))
                tool_s += secs
                if tool.group(1) != "bash":
                    continue
                bash_n += 1
                bash_s += secs
                try:
                    command = json.loads(tool.group(2)).get("command", "")
                except ValueError:
                    continue
                heads = [head_word(s) for s in SEGMENT.split(command) if s.strip()]
                if not any(h in SEARCHERS for h in heads):
                    continue
                if all(h in READ_ONLY for h in heads):
                    search_n += 1
                    search_s += secs
                    latencies.append(secs)
                else:
                    compound_n += 1
                    compound_s += secs

    print(f"model decode          {decode_s:10.1f} s   over {steps} steps")
    print(f"all tool execution    {tool_s:10.1f} s")
    print(f"  bash               {bash_s:10.1f} s   {bash_n} calls")
    print(f"  search-only bash   {search_s:10.1f} s   {search_n} calls   "
          f"mean {1000 * search_s / max(search_n, 1):.1f} ms")
    print(f"  grep in a compound {compound_s:10.1f} s   {compound_n} calls   "
          "(a build or test run, not search)")
    if latencies:
        ordered = sorted(latencies)
        pick = lambda f: ordered[min(len(ordered) - 1, int(f * len(ordered)))]  # noqa: E731
        print(f"\nsearch latency        p50 {1000 * pick(0.5):.0f} ms   "
              f"p90 {1000 * pick(0.9):.0f} ms   max {1000 * ordered[-1]:.0f} ms")
    total = decode_s + tool_s
    if total:
        print(f"\nsearch share of chad's wall clock: {100 * search_s / total:.4f}%")
        for speedup in (2.0, 3.0, 10.0):
            saved = search_s - search_s / speedup
            print(f"  a {speedup:4.1f}x faster grep would return {saved:6.1f} s "
                  f"of {total:.0f} s  ({100 * saved / total:.4f}%)")


def corpus_reuse() -> None:
    counts: dict[str, int] = {}
    for path in sorted(glob.glob(SESSIONS)):
        if path.endswith("index.json"):
            continue
        try:
            with open(path) as fh:
                cwd = json.load(fh).get("cwd", "")
        except (OSError, ValueError):
            continue
        if cwd:
            counts[cwd] = counts.get(cwd, 0) + 1
    if not counts:
        return
    sessions = sum(counts.values())
    print(f"\n{sessions} sessions across {len(counts)} distinct working directories "
          f"({sessions / len(counts):.2f} sessions per directory)")
    print(f"{'sessions':>8}  {'files':>7}  cwd")
    for cwd, n in sorted(counts.items(), key=lambda kv: -kv[1])[:6]:
        if os.path.isdir(cwd):
            listed = subprocess.run(["rg", "--files"], cwd=cwd, capture_output=True,
                                    stdin=subprocess.DEVNULL).stdout.splitlines()
            size = str(len(listed))
        else:
            size = "gone"
        print(f"{n:8}  {size:>7}  {cwd}")


def main() -> None:
    wall_clock()
    corpus_reuse()


main()
