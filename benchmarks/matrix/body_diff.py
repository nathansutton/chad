"""Where does the byte-stable prefix of two consecutive requests end?

llama-server's prefix cache matches the rendered prompt token by token from the front.
The proxy's `sys_sha`/`tools_sha` say whether the system message or the tool list
changed between turns; they cannot say anything about the conversation body, and a
harness that re-serialises earlier messages differently each turn defeats the cache
just as thoroughly as one that rewrites its system prompt. This is the instrument for
that case. Give it two request bodies captured by `sampler_proxy.py --bodies` (or
`run.py ... --capture-bodies`) and it reports, for the parts of the request the server
actually renders:

  - whether the system message and the tool list are byte-identical
  - the first message index at which the two `messages` arrays differ, and how
  - the shared prefix of the two arrays rendered as JSON, in characters — a proxy for
    how far the cache could have matched, engine-independent

It is a diagnostic, not a metric: run it on the pair of turns whose `cache_n` fell,
and read the output.

    uv run python benchmarks/matrix/body_diff.py _runs/bodies/goose+llama/wordy-0/00012.json \\
                                                 _runs/bodies/goose+llama/wordy-0/00013.json
    uv run python benchmarks/matrix/body_diff.py --all _runs/bodies/goose+llama/wordy-0/
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _messages(doc: dict) -> list:
    m = doc.get("messages")
    if m is None and isinstance(doc.get("input"), list):
        m = doc["input"]
    return m if isinstance(m, list) else []


def _dump(x) -> str:
    return json.dumps(x, sort_keys=True, ensure_ascii=False)


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def compare(prev: dict, cur: dict) -> dict:
    """The structured verdict for one consecutive pair."""
    pm, cm = _messages(prev), _messages(cur)
    out = {"n_messages": (len(pm), len(cm)),
           "system_same": _dump(prev.get("instructions") or next(
               (m for m in pm if isinstance(m, dict) and m.get("role") in ("system", "developer")), None))
           == _dump(cur.get("instructions") or next(
               (m for m in cm if isinstance(m, dict) and m.get("role") in ("system", "developer")), None)),
           "tools_same": _dump(prev.get("tools")) == _dump(cur.get("tools")),
           "first_diff_index": None, "first_diff": None}
    for i, (a, b) in enumerate(zip(pm, cm)):
        if _dump(a) != _dump(b):
            out["first_diff_index"] = i
            da, db = _dump(a), _dump(b)
            k = _common_prefix(da, db)
            out["first_diff"] = {"role": (a.get("role") if isinstance(a, dict) else None,
                                          b.get("role") if isinstance(b, dict) else None),
                                 "prev": da[max(0, k - 60):k + 100],
                                 "cur": db[max(0, k - 60):k + 100]}
            break
    else:
        # No differing message within the shared length: the new request only appended.
        out["first_diff_index"] = min(len(pm), len(cm)) if len(pm) != len(cm) else None
    out["shared_prefix_chars"] = _common_prefix(_dump(pm), _dump(cm))
    out["prev_chars"] = len(_dump(pm))
    out["appended_only"] = out["first_diff"] is None and len(cm) >= len(pm)
    return out


def render(name_a: str, name_b: str, v: dict) -> str:
    lines = [f"{os.path.basename(name_a)} -> {os.path.basename(name_b)}: "
             f"messages {v['n_messages'][0]} -> {v['n_messages'][1]}, "
             f"system {'same' if v['system_same'] else 'CHANGED'}, "
             f"tools {'same' if v['tools_same'] else 'CHANGED'}, "
             f"shared prefix {v['shared_prefix_chars']:,}/{v['prev_chars']:,} chars"]
    if v["appended_only"]:
        lines.append("  earlier messages byte-identical; the new request only appended "
                     "-> the cache could match everything the previous turn held")
    else:
        d = v["first_diff"]
        lines.append(f"  first differing message: index {v['first_diff_index']} "
                     f"(role {d['role'][0]} -> {d['role'][1]})")
        lines.append(f"    prev: …{d['prev']}…")
        lines.append(f"    cur : …{d['cur']}…")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="two body files, or with --all one directory")
    ap.add_argument("--all", action="store_true",
                    help="every consecutive pair in the directory, in capture order")
    a = ap.parse_args(argv)
    if a.all:
        d = a.paths[0]
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))
    else:
        files = a.paths
    if len(files) < 2:
        print("need at least two bodies", file=sys.stderr)
        return 2
    docs = [json.load(open(f)) for f in files]
    for (fa, da), (fb, db) in zip(zip(files, docs), zip(files[1:], docs[1:])):
        print(render(fa, fb, compare(da, db)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
