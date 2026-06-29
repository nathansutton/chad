"""Context compaction for long agentic sessions (extracted from agent.py).

Operates on a plain `messages` list plus two callbacks — `render()` (returns the
current prompt token ids) and `emit(kind, text)` (status/info display) — so it
carries no Agent state. `Agent` calls `compact_if_needed(self.messages, self._render,
self._emit, self.ctx_limit, prompt_ids)`; the `compact_now` (/compact) path reuses
`_headtail`/`_COLLAPSED` via the Agent aliases. Behavior is byte-identical to the old
in-class method — same passes, same thresholds (0.7 target, keep-last-N), same order.
"""

from . import skills

_COLLAPSED = "[…earlier output trimmed to save context…]"


def _headtail(text: str, head: int = 12, tail: int = 8, max_chars: int = 8000) -> str:
    """Keep the first/last few lines of a long output; drop the middle. The
    head and tail carry the most signal (what a command was / how it ended),
    and this preserves far more than a bare stub while still reclaiming space.
    A char cap also clips pathological single-line blobs (minified files etc.)
    that have too few newlines for line-based trimming to help."""
    lines = text.splitlines()
    if len(lines) > head + tail + 3:
        text = "\n".join(
            lines[:head]
            + [f"  {_COLLAPSED} ({len(lines) - head - tail} lines)"]
            + lines[-tail:])
    if len(text) > max_chars:
        keep = max_chars // 2
        text = text[:keep] + f"\n  {_COLLAPSED}\n" + text[-keep:]
    return text


def compact_if_needed(messages, render, emit, ctx_limit, prompt_ids):
    """Context compaction for long agentic sessions. On a non-trimmable cache
    (Ornith) any prefix change forces a full re-prefill, so compaction is
    expensive — we must reclaim enough in ONE pass that it won't re-trigger next
    step (otherwise: full re-prefill every step). We therefore reclaim down to a
    target well below the limit, escalating until we get there:
      1. strip stale <think> reasoning from older assistant turns;
      2. head/tail-truncate the oldest large tool outputs;
      3. as a last resort, drop the oldest messages entirely.
    Recent context (last few tool results / assistant turns) is kept verbatim."""
    if len(prompt_ids) <= ctx_limit:
        return prompt_ids
    target = int(ctx_limit * 0.7)  # reclaim to here so we don't recompact soon
    emit("status", "Compacting context")
    before = len(prompt_ids)

    def cur_len():
        return len(render())

    def trunc_tools(keep_recent, head, tail, max_chars):
        # Activated skill instructions are durable behavioral guidance — exempt them
        # from truncation (silently shrinking them degrades the agent with no error).
        idxs = [i for i, m in enumerate(messages)
                if m.get("role") == "tool" and _COLLAPSED not in m["content"]
                and not skills.is_skill_message(m)]
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
    # NEVER delete the system prompt or the most recent real user query (the
    # active task) and everything after it: removing the active query makes the
    # chat template raise "No user query found in messages" and crashes the turn.
    def _last_user_idx():
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].get("role") == "user":
                return i
        return None
    BATCH = 8  # delete up to this many oldest messages between budget re-checks
    guard = 0
    while cur_len() > target and guard < 500:
        lu = _last_user_idx()
        ceil = lu if lu is not None else len(messages)
        # Oldest deletable indices: after the system prompt, before the active user query,
        # and never an activated skill message (durable guidance is kept verbatim). Collect
        # up to BATCH of them in one scan so we re-tokenize (cur_len) once per batch instead
        # of once per deletion — render() tokenizes the whole transcript, so per-message
        # re-checks made pass 3 O(N²).
        victims = [i for i in range(1, ceil)
                   if not skills.is_skill_message(messages[i])][:BATCH]
        if not victims:  # nothing left to shed without touching protected content
            break
        for i in reversed(victims):  # delete high→low so earlier indices stay valid
            del messages[i]
        guard += 1
    # pass 4: last resort — recent outputs alone still exceed target; truncate
    # everything that's left, hard.
    if cur_len() > target:
        trunc_tools(keep_recent=0, head=6, tail=4, max_chars=2000)

    new_ids = render()
    emit("info", f"  [compacted context: {before}→{len(new_ids)} tokens]")
    return new_ids
