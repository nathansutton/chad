"""Tests for context compaction (compaction.py) — previously ZERO coverage.

compaction.py is a crown-jewel path: on Ornith's non-trimmable cache any prefix
change forces a full re-prefill, so a compaction bug is both likely (multi-pass
heuristics) and expensive (re-prefill every step). The module's own docstring makes
two load-bearing claims that nothing verified until now:

  1. `_headtail` keeps head+tail, drops the middle, and hard-caps total chars.
  2. The drop-oldest pass NEVER deletes the system prompt or the active user query
     (the most recent `user` message) — doing so makes the chat template raise
     "No user query found in messages" and crashes the turn.

These are pure-ish (messages list + two callbacks), so we drive them directly with
a fake token-renderer instead of loading a model. No model, runs in the fast gate.

  messages                      compact_if_needed                  invariant
  ┌─ system ────────┐  prompt over ctx_limit?  ──no──▶ return as-is
  │  user (old)      │           │yes
  │  tool (huge) ×N  │  pass1 strip <think>
  │  assistant       │  pass2 head/tail tools  ──▶  shrinks toward 0.7*limit
  │  user (ACTIVE) ◀─┼─ pass3 drop oldest, but NEVER cross this line or system[0]
  │  tool (recent)   │  pass4 hard-truncate remainder
  └─────────────────┘
"""
from chad.compaction import _COLLAPSED, _headtail, compact_if_needed

passed = 0
failed = 0


def check(desc, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS: {desc}")
    else:
        failed += 1
        raise AssertionError(f"{desc}{(' — ' + detail) if detail else ''}")


# A render() stand-in: token ids ≈ characters (1 char -> 1 "token"). Deterministic
# and monotonic in message size, which is all the pass logic actually needs.
def make_render(messages):
    def render():
        return list("".join(m["content"] for m in messages))
    return render


def noop_emit(kind, text):
    pass



def test_compaction():
    # === _headtail (pure) =======================================================

    # line-based middle drop
    long_lines = "\n".join(f"line {i}" for i in range(100))
    ht = _headtail(long_lines, head=12, tail=8)
    check("_headtail keeps head lines", "line 0" in ht and "line 11" in ht)
    check("_headtail keeps tail lines", "line 99" in ht and "line 92" in ht)
    check("_headtail drops the middle", "line 50" not in ht)
    check("_headtail inserts collapse marker", _COLLAPSED in ht)
    check("_headtail reports dropped count", "(80 lines)" in ht,
          f"got: {ht!r}")

    # short input is returned untouched (not enough lines to trim)
    short = "line 1\nline 2\nline 3"
    check("_headtail leaves short text unchanged", _headtail(short) == short)

    # char cap clips a pathological single-line blob (too few newlines to line-trim)
    blob = "x" * 20000
    ht_blob = _headtail(blob, max_chars=8000)
    check("_headtail char-caps a giant single line", len(ht_blob) < 9000,
          f"len={len(ht_blob)}")
    check("_headtail char-cap keeps head and tail bytes",
          ht_blob.startswith("x") and ht_blob.endswith("x") and _COLLAPSED in ht_blob)


    # === compact_if_needed: no-op below the limit ===============================

    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "do the thing"},
    ]
    ids = make_render(msgs)()
    out = compact_if_needed(msgs, make_render(msgs), noop_emit, ctx_limit=10_000, prompt_ids=ids)
    check("under ctx_limit: returns prompt_ids unchanged", out == ids)
    check("under ctx_limit: messages untouched", len(msgs) == 2)


    # === compact_if_needed: strips old <think> reasoning (pass 1) ===============

    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "<think>" + ("reason " * 200) + "</think>ANSWER-1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "<think>" + ("more " * 200) + "</think>ANSWER-2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "<think>" + ("again " * 200) + "</think>ANSWER-3"},
        {"role": "user", "content": "q4 ACTIVE"},
    ]
    render = make_render(msgs)
    before = len(render())
    compact_if_needed(msgs, render, noop_emit, ctx_limit=before - 1, prompt_ids=render())
    # pass1 keeps the last 2 assistant turns' thinking; the OLDEST (index 2) is stripped,
    # while the two recent thinking turns (indices 4, 6) are preserved verbatim.
    check("pass1 strips OLDEST <think> reasoning",
          "<think>" not in msgs[2]["content"] and "ANSWER-1" in msgs[2]["content"],
          f"got: {msgs[2]['content'][:40]!r}")
    check("pass1 preserves the 2 most-recent thinking turns",
          "<think>" in msgs[4]["content"] and "<think>" in msgs[6]["content"])


    # === INVARIANT: never delete system[0] or the active (last) user query ======

    # Build a transcript far over budget with many droppable old tool outputs, so the
    # drop-oldest loop (pass 3) is forced to run hard.
    msgs = [{"role": "system", "content": "SYSTEM-PROMPT"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"old-user-{i}"})
        msgs.append({"role": "tool", "content": f"old-tool-output-{i} " * 50})
    msgs.append({"role": "user", "content": "ACTIVE-QUERY-DO-NOT-DELETE"})
    msgs.append({"role": "tool", "content": "recent-tool-result"})

    render = make_render(msgs)
    big = len(render())
    compact_if_needed(msgs, render, noop_emit, ctx_limit=big // 4, prompt_ids=render())

    check("INVARIANT: system prompt survives compaction",
          msgs[0]["role"] == "system" and msgs[0]["content"] == "SYSTEM-PROMPT")
    check("INVARIANT: active user query survives compaction",
          any(m["content"] == "ACTIVE-QUERY-DO-NOT-DELETE" for m in msgs),
          "active query was deleted — chat template would crash with 'No user query found'")
    check("INVARIANT: content after the active query survives",
          any(m["content"] == "recent-tool-result" for m in msgs))
    check("compaction actually reclaimed space",
          len(render()) < big, f"before={big} after={len(render())}")


    # === pass 3 batches renders (does NOT re-tokenize per deleted message) ======

    # Same shape as the INVARIANT fixture (≈20 user/tool pairs, far over budget so
    # pass 3 drops many messages). Wrap render() in a counter and assert it is called
    # far fewer times than the number of messages deleted — proving the loop batches.
    msgs = [{"role": "system", "content": "SYSTEM-PROMPT"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"old-user-{i}"})
        msgs.append({"role": "tool", "content": f"old-tool-output-{i} " * 50})
    msgs.append({"role": "user", "content": "ACTIVE-QUERY-DO-NOT-DELETE"})
    msgs.append({"role": "tool", "content": "recent-tool-result"})

    calls = {"n": 0}
    base = make_render(msgs)

    def counting_render():
        calls["n"] += 1
        return base()

    n_before = len(msgs)
    compact_if_needed(msgs, counting_render, noop_emit,
                      ctx_limit=len(base()) // 4, prompt_ids=base())
    deleted = n_before - len(msgs)
    check("pass3 batches renders (not one per deleted message)",
          deleted > 8 and calls["n"] < deleted,
          f"deleted={deleted} renders={calls['n']}")


    # === degenerate: nothing droppable between system and the active query ======

    # system + active query only, but reported over budget: the guard must break out
    # (ceil-1 < 2) instead of deleting the system prompt or looping 500×.
    msgs = [
        {"role": "system", "content": "S" * 100},
        {"role": "user", "content": "ACTIVE"},
    ]
    render = make_render(msgs)
    out = compact_if_needed(msgs, render, noop_emit, ctx_limit=1, prompt_ids=render())
    check("degenerate: system + active query both survive (no crash, no over-delete)",
          len(msgs) == 2 and msgs[0]["role"] == "system"
          and msgs[1]["content"] == "ACTIVE")



if __name__ == "__main__":
    test_compaction()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
