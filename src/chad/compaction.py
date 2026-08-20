"""Context compaction for long agentic sessions (extracted from agent.py).

Operates on a plain `messages` list plus two callbacks — `render()` (returns the
current prompt token ids) and `emit(kind, text)` (status/info display) — so it
carries no Agent state. `Agent` calls `compact_if_needed(self.messages, self._render,
self._emit, self.ctx_limit, prompt_ids)`; the `compact_now` (/compact) path reuses
`_headtail`/`_COLLAPSED` via the Agent aliases.

Every pass marks what it trimmed with `_COLLAPSED`, in-band, so a clipped
result never reads as complete — and the full original goes to a spill file whose
path rides in the same message, so the marker is a pointer and not just an apology.
"""

import logging

from . import levers, skills, spill

log = logging.getLogger("chad")

_COLLAPSED = "[…earlier output trimmed to save context…]"


def _pointer(original: str) -> str:
    """The trailing line that makes a trim recoverable: `\n[… full output saved to
    <path> …]`, or "" if there is nowhere to point.

    Compaction is the one path in the harness that used to destroy bytes outright —
    bash has spilled its omitted middle since it shipped, but a result the compactor
    head/tail-trimmed was simply gone, and the model's only route back was to issue
    the identical call again (measured over the banked archive: 7.6% of trimmed
    results, each costing a dead turn plus a full re-prefill). A result that already
    carries a spill pointer reuses it rather than writing the same body twice."""
    if not levers.enabled("trim_spill"):
        return ""
    path = spill.path_in(original) or spill.write(original, "compact")
    if not path:
        return ""
    levers.fired("trim_spill", chars=len(original))
    return (f"\n[… full output saved to {path} — grep/sed that file instead of "
            f"re-running the call that produced it …]")


def _headtail(text: str, head: int = 12, tail: int = 8, max_chars: int = 8000) -> str:
    """Keep the first/last few lines of a long output; drop the middle. The
    head and tail carry the most signal (what a command was / how it ended),
    and this preserves far more than a bare stub while still reclaiming space.
    A char cap also clips pathological single-line blobs (minified files etc.)
    that have too few newlines for line-based trimming to help.

    Whatever is dropped goes to a spill file first and the trimmed text names it, so
    the middle is a `grep` away rather than a re-run away."""
    original = text
    lines = text.splitlines()
    if len(lines) > head + tail + 3:
        text = "\n".join(
            lines[:head]
            + [f"  {_COLLAPSED} ({len(lines) - head - tail} lines)"]
            + lines[-tail:])
    if len(text) > max_chars:
        keep = max_chars // 2
        text = text[:keep] + f"\n  {_COLLAPSED}\n" + text[-keep:]
    if text is original:  # nothing dropped — no loan to record
        return text
    return text + _pointer(original)


# When compaction cannot get under ctx_limit (the protected floor — system prompt +
# skills + recent window + active query — exceeds the limit), re-running it every step
# is pure loss: each attempt mutates the transcript, which diverges the prefix and
# forces a FULL re-prefill on the non-trimmable cache (measured: 26-28s per step, 7
# steps consecutive, traces/session2.jsonl). The latch skips re-compaction until the
# transcript has actually grown this many tokens past the recorded floor; the engine's
# memory clamps make the small over-limit excursion safe.
_OVERLIMIT_REARM = 2048


def compact_if_needed(messages, render, emit, ctx_limit, prompt_ids, state=None):
    """Context compaction for long agentic sessions. On a non-trimmable cache
    any prefix change forces a full re-prefill, so compaction is
    expensive — we must reclaim enough in ONE pass that it won't re-trigger next
    step (otherwise: full re-prefill every step). We therefore reclaim down to a
    target well below the limit, escalating until we get there:
      1. strip stale <think> reasoning from older assistant turns;
      2. head/tail-truncate the oldest large tool outputs;
      3. as a last resort, drop the oldest messages entirely.
    Recent context (last few tool results / assistant turns) is kept verbatim.

    `state`, if given, is a caller-owned dict that persists across calls; it holds
    the over-limit latch (see _OVERLIMIT_REARM) so a floor-bound transcript is not
    re-compacted — and its warm cache not destroyed — every single step."""
    if len(prompt_ids) <= ctx_limit:
        if state is not None:
            state.pop("overlimit_floor", None)  # back under: disarm the latch
        return prompt_ids
    if state is not None:
        floor = state.get("overlimit_floor")
        if floor and len(prompt_ids) <= floor + _OVERLIMIT_REARM:
            # Floor-bound: last compaction couldn't reclaim below ctx_limit and the
            # transcript hasn't grown enough for a retry to fare any better. Leave
            # it untouched so the prefix cache stays warm.
            return prompt_ids
    target = int(ctx_limit * 0.7)  # reclaim to here so we don't recompact soon
    emit("status", "Compacting context")
    before = len(prompt_ids)

    def cur_len():
        return len(render())

    def trunc_tools(keep_recent, head, tail, max_chars):
        # Only tool results are truncated here; a loaded skill now rides as a USER turn
        # (skills.load), so it is out of this pass's reach by construction and needs no
        # exemption. Pass 3 is where it has to be protected from deletion.
        idxs = [i for i, m in enumerate(messages)
                if m.get("role") == "tool" and _COLLAPSED not in m["content"]]
        stop = len(idxs) - keep_recent if keep_recent else len(idxs)
        for i in idxs[:stop]:
            if len(messages[i]["content"]) > 400:
                messages[i]["content"] = _headtail(
                    messages[i]["content"], head, tail, max_chars)

    # pass 1: drop old reasoning (keep the last 2 assistant turns' thinking)
    for i in [i for i, m in enumerate(messages)
              if m.get("role") == "assistant" and "</think>" in m["content"]][:-2]:
        c = messages[i]["content"]
        messages[i]["content"] = c.split("</think>", 1)[1].lstrip("\n")

    # pass 2: head/tail-truncate older tool outputs (keep the last 4 verbatim)
    if cur_len() > target:
        trunc_tools(keep_recent=4, head=12, tail=8, max_chars=8000)
    # pass 3: drop the oldest messages outright (keep system + recent window) —
    # old context is less useful than recent, so shed it before touching recent.
    # NEVER delete the system prompt or the active user query: removing the query
    # makes the chat template raise "No user query found in messages" and crashes
    # the turn. The protected span is those two plus the last KEEP_RECENT messages
    # — NOT "everything after the last user query", which in a one-user-turn
    # headless run collapses the deletable range to nothing and makes this pass
    # dead code. Bounding by a trailing window instead reclaims from old churn
    # while keeping the live working set verbatim.
    def _last_user_idx():
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].get("role") == "user":
                return i
        return None
    KEEP_RECENT = 6  # trailing messages held verbatim (the live working set)
    BATCH = 8  # delete up to this many oldest messages between budget re-checks
    guard = 0
    while cur_len() > target and guard < 500:
        lu = _last_user_idx()
        ceil = max(1, len(messages) - KEEP_RECENT)
        # Oldest deletable indices: after the system prompt, before the recent window,
        # never the active user query, and never a loaded skill's turn (the user asked
        # for that guidance by name; dropping it degrades the agent with no error, and
        # nothing would reload it). Collect up to BATCH of them in one scan so we
        # re-tokenize (cur_len) once per batch instead of once per deletion — render()
        # tokenizes the whole transcript, so per-message re-checks made pass 3 O(N²).
        victims = [i for i in range(1, ceil)
                   if i != lu and not skills.is_skill_message(messages[i])][:BATCH]
        if not victims:  # nothing left to shed without touching protected content
            break
        for i in reversed(victims):  # delete high→low so earlier indices stay valid
            del messages[i]
        guard += 1
    # pass 4: last resort — recent outputs alone still exceed target. Truncate what
    # is left oldest-first, escalating only as far as the budget demands, and NEVER
    # touching the newest tool result: compaction runs at the top of a step, before
    # generation, so the newest result is the one the model is about to read. The old
    # keep_recent=0 shredded it unread — the agent would search, lose the hits to
    # compaction, and search again with nothing gained.
    # trunc_tools skips already-collapsed messages, so each rung only bites into the
    # newer results the previous rung spared.
    for keep, head, tail, mx in ((4, 8, 6, 4000), (2, 6, 4, 2000), (1, 4, 3, 1000)):
        if cur_len() <= target:
            break
        trunc_tools(keep_recent=keep, head=head, tail=tail, max_chars=mx)
    # Absolute last resort: the newest result ALONE still blows the hard limit — a single
    # huge file read, which no amount of shedding older context can offset. Sparing it is
    # a heuristic; staying under ctx_limit is not, since returning over-limit leaves the
    # caller re-prefilling a context that only grows. Clip it head/tail like any other.
    if cur_len() > ctx_limit:
        trunc_tools(keep_recent=0, head=6, tail=4, max_chars=2000)

    new_ids = render()
    # Log, not just emit: emit() only reaches the TUI, so a compaction that failed to
    # converge left no trace at all. `over_limit` is the one to alert on — it means the next step re-prefills a
    # context that only grows.
    log.info("COMPACT %d→%d tokens | limit=%d target=%d msgs=%d%s",
             before, len(new_ids), ctx_limit, target, len(messages),
             " OVER-LIMIT" if len(new_ids) > ctx_limit else "")
    if state is not None:
        # Latch on RUNWAY, not merely on "did we get under the limit". The cost of a
        # compaction is fixed and large — the whole transcript past the stable prefix
        # gets re-prefilled — while the benefit is only the headroom it buys before the
        # next one. Landing a hair under the limit is therefore nearly as bad as landing
        # over it: the next step re-crosses and pays the full re-prefill again. Measured
        # on a completed ky session, where the floor rises as the protected recent window
        # fills, so each successive pass reclaims less and costs more:
        #   freed 9777 / re-prefilled 9069  (0.9x)
        #   freed 5325 / re-prefilled 13552 (2.5x)
        #   freed 2720 / re-prefilled 15984 (5.9x)
        #   freed 1547 / re-prefilled 16948 (11x)
        #   freed  884 / re-prefilled 17667 (20x)
        # 73,220 tokens re-prefilled to free 20,253 — 43% of that session's wall clock.
        # Only the first pass paid for itself; the latch below stops the treadmill after
        # it. Same excursion-safety argument as the over-limit case: the engine's memory
        # clamps cover the small overshoot.
        runway = ctx_limit - len(new_ids)
        if runway < _OVERLIMIT_REARM:
            if not state.get("overlimit_floor"):
                log.warning("COMPACT low-yield (%d→%d, %d tokens of runway): latching "
                            "until context grows past %d (+%d) rather than re-prefilling "
                            "%d tokens again next step",
                            before, len(new_ids), runway, len(new_ids),
                            _OVERLIMIT_REARM, len(new_ids))
            state["overlimit_floor"] = len(new_ids)
        else:
            state.pop("overlimit_floor", None)
    emit("info", f"  [compacted context: {before}→{len(new_ids)} tokens]")
    return new_ids
