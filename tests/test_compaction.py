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



def test_subagent_compaction():
    """A sub-agent's transcript has exactly ONE user message — the task prompt at index
    1, since run_turn appends the prompt once and every nudge lands as a `tool` message.

    That shape used to disable pass 3 entirely: its deletable range was
    `range(1, last_user_idx)` == `range(1, 1)` == empty, so a sub-agent fell straight
    through to pass 4, which ran with keep_recent=0 and truncated *every* tool result —
    including the one that had just arrived. Compaction runs at the top of a step, before
    generation, so the sub-agent lost each grep's hits before it could read them: it would
    search, compact, search again, and hit its step cap with nothing to return.

    Both invariants are asserted below against the real compact_if_needed."""

    def sub_transcript(n_searches, hits=140):
        """system + task prompt + n rounds of (assistant, big grep result)."""
        msgs = [{"role": "system", "content": "S" * 3000},
                {"role": "user", "content": "Find where retry/backoff is handled."}]
        for i in range(n_searches):
            msgs.append({"role": "assistant", "content": f"<think>reason {i}</think>\ngrep"})
            msgs.append({"role": "tool", "name": "grep", "content": "\n".join(
                f"src/mod{i}/file{j}.py:{j}: def handle_retry_{j}() " + "x" * 55
                for j in range(hits))})
        return msgs

    # --- pass 3 must actually shed old churn in a sub-agent ----------------------
    msgs = sub_transcript(6)
    n_before = len(msgs)
    render = make_render(msgs)
    ctx = len(render()) // 2
    compact_if_needed(msgs, render, noop_emit, ctx_limit=ctx, prompt_ids=render())
    check("sub-agent: pass 3 drops oldest messages (not dead code)",
          len(msgs) < n_before, f"msgs stayed at {len(msgs)}")
    check("sub-agent: reclaims to under ctx_limit", len(render()) <= ctx,
          f"{len(render())} > {ctx}")
    check("sub-agent: system prompt survives", msgs[0]["role"] == "system")
    check("sub-agent: task prompt survives (chat template needs a user query)",
          any(m["role"] == "user" for m in msgs))

    # --- the newest tool result is never shredded before the model reads it -------
    # Hold the message OBJECT, not its index: pass 3 now deletes, so indices shift.
    msgs = sub_transcript(5)
    freshest = msgs[-1]
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit,
                      ctx_limit=len(render()) // 2, prompt_ids=render())
    check("sub-agent: freshest tool result survives compaction intact",
          _COLLAPSED not in freshest["content"],
          "the result the model is about to read was truncated unread")
    check("sub-agent: freshest tool result is still in the transcript",
          any(m is freshest for m in msgs))

    # --- a sustained search loop keeps every fresh result, over many compactions --
    # The dogfooding symptom: "searches, compacts 3-4 times, returns nothing."
    msgs = sub_transcript(0)
    render = make_render(msgs)
    ctx, compactions, shredded = 16000 * 4, 0, 0  # *4: this render is chars-per-token
    for step in range(12):  # SUBAGENT_MAX_STEPS-scale run
        msgs.append({"role": "assistant", "content": f"<think>t{step}</think>\ngrep"})
        fresh = {"role": "tool", "name": "grep", "content": "\n".join(
            f"src/s{step}/f{j}.py:{j}: hit {j} " + "x" * 55 for j in range(140))}
        msgs.append(fresh)
        pre = len(render())
        compact_if_needed(msgs, render, noop_emit, ctx_limit=ctx, prompt_ids=render())
        if len(render()) < pre:
            compactions += 1
        if _COLLAPSED in fresh["content"]:
            shredded += 1
        check(f"sub-agent loop step {step}: stays under ctx_limit", len(render()) <= ctx)
    check("sub-agent loop: compaction actually fired", compactions >= 2,
          f"only {compactions} compactions — test no longer exercises the path")
    check("sub-agent loop: no fresh search result was ever shredded unread",
          shredded == 0, f"{shredded} of 12 steps lost their freshest result")

    # --- sparing the newest result must never cost convergence --------------------
    # Caught by a live sub-agent run, not by the synthetic loop above: when one fresh
    # read is itself larger than target, no amount of shedding older context gets under
    # the limit. Staying under ctx_limit is mandatory (an over-limit return just grows
    # the context every step); sparing the newest result is only a heuristic. The hard
    # ceiling wins, and the newest result gets clipped head/tail like any other.
    msgs = sub_transcript(1)
    msgs.append({"role": "assistant", "content": "reading the whole file"})
    msgs.append({"role": "tool", "name": "read",
                 "content": "\n".join(f"{i}: line of a very large file " + "y" * 70
                                      for i in range(4000))})
    render = make_render(msgs)
    # Hold the OBJECT, not the index: compaction now appends an in-band notice after the
    # passes run, so the big read is no longer the tail message.
    big_read = msgs[-1]
    ctx = len(big_read["content"]) // 2   # newest message alone exceeds the limit
    compact_if_needed(msgs, render, noop_emit, ctx_limit=ctx, prompt_ids=render())
    check("oversized newest result: still converges under ctx_limit",
          len(render()) <= ctx, f"{len(render())} > {ctx} — compaction did not converge")
    check("oversized newest result: clipped head/tail, not dropped",
          _COLLAPSED in big_read["content"] and big_read["content"].startswith("0: line"))


def test_overlimit_latch():
    """When compaction CANNOT reclaim below ctx_limit (the protected
    floor — system + skills + recent window + active query — exceeds the window),
    re-running it every step destroys the warm prefix cache for nothing: each attempt
    mutates the transcript, which diverges the prefix and forces a full re-prefill on
    the non-trimmable cache (measured 26-28s/step, traces/session2.jsonl). With a
    caller-owned `state` dict, the over-limit result latches a floor; subsequent calls
    return the prompt untouched until the transcript grows _OVERLIMIT_REARM tokens
    past it, and the latch disarms the moment the prompt is back under the limit."""
    from chad.compaction import _OVERLIMIT_REARM

    # A transcript whose PROTECTED span alone exceeds ctx_limit: a big system prompt,
    # an active user query, and recent messages inside the KEEP_RECENT window. Every
    # message is protected, so the passes can reclaim nothing meaningful.
    def floor_bound_msgs():
        return ([{"role": "system", "content": "S" * 4000}]
                + [{"role": "user", "content": "the active query"}]
                + [{"role": "tool", "name": "read", "content": f"recent {i} " + "r" * 100}
                   for i in range(4)])

    msgs = floor_bound_msgs()
    render = make_render(msgs)
    ctx = 2000  # far below the ~4500-char protected floor
    state = {}

    out1 = compact_if_needed(msgs, render, noop_emit, ctx, render(), state=state)
    check("floor-bound compaction returns over-limit", len(out1) > ctx, len(out1))
    check("over-limit latches a floor in state", state.get("overlimit_floor"), state)

    # Second call, transcript unchanged: the latch must skip all work — messages
    # untouched (same object contents) and the prompt returned as-is.
    snapshot = [dict(m) for m in msgs]
    out2 = compact_if_needed(msgs, render, noop_emit, ctx, render(), state=state)
    check("latched: prompt returned unchanged", out2 == render(), len(out2))
    check("latched: transcript not mutated",
          [dict(m) for m in msgs] == snapshot)

    # Growth past the rearm margin re-attempts compaction (which mutates again).
    msgs.append({"role": "tool", "name": "read",
                 "content": "g" * (_OVERLIMIT_REARM + 200)})
    before = [dict(m) for m in msgs]
    compact_if_needed(msgs, render, noop_emit, ctx, render(), state=state)
    check("rearm: growth past the margin re-compacts",
          [dict(m) for m in msgs] != before)

    # Back under the limit: the latch disarms.
    small = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    r2 = make_render(small)
    compact_if_needed(small, r2, noop_emit, ctx, r2(), state=state)
    check("under-limit call disarms the latch", "overlimit_floor" not in state, state)

    # And with NO state (legacy callers, sub-agents): behavior is unchanged — every
    # over-limit call recompacts. Just prove it doesn't crash and stays over-limit.
    msgs3 = floor_bound_msgs()
    r3 = make_render(msgs3)
    out = compact_if_needed(msgs3, r3, noop_emit, ctx, r3())
    check("stateless call still works (legacy path)", len(out) > ctx, len(out))


if __name__ == "__main__":
    test_compaction()
    test_subagent_compaction()
    test_overlimit_latch()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
