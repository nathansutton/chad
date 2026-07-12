"""Context compaction for long agentic sessions (extracted from agent.py).

Operates on a plain `messages` list plus two callbacks — `render()` (returns the
current prompt token ids) and `emit(kind, text)` (status/info display) — so it
carries no Agent state. `Agent` calls `compact_if_needed(self.messages, self._render,
self._emit, self.ctx_limit, prompt_ids)`; the `compact_now` (/compact) path reuses
`_headtail`/`_COLLAPSED` via the Agent aliases.

Compaction used to be silent to the model: passes 1-4 shrink the transcript in place,
`emit()` prints a line to the TUI, and the model's next turn sees a head/tail-clipped
tool result with no indication that anything is missing. The predictable failure is that
it answers from the clipped text as though it had read the whole thing. Two additions
close that, each behind a lever (see levers.py):

  compact_offload — spill the untrimmed text to a file before shrinking, so the content
                    is recoverable rather than destroyed;
  compact_notice  — append an in-band message naming what was lost and where it went.

The notice rides at the TAIL of the transcript, next to the data the model is about to
read, rather than in the system prompt where a standing rule washes out. Placement is
the whole trick: the same words in the tool *description* of `read` did nothing until
they moved into what `read` *returned*.

The cost objection does not apply here. Ornith's hybrid SSM/attention cache is
non-trimmable, so compaction already forces a full re-prefill; appending a message at
that exact moment is free. We are paying the prefill either way.
"""

import logging
import os
import time

from . import levers, skills

log = logging.getLogger("chad")

_COLLAPSED = "[…earlier output trimmed to save context…]"

# Marks the injected notice so the next compaction can strip the stale one instead of
# stacking a new copy every time the window fills.
_NOTICE_TAG = "[context-compacted]"

# chad has already put a hole in this foot once: an unbounded ~/.cache/chad/kv filled the
# disk and wedged Docker. An offload file grows by every byte compaction reclaims, on a
# loop that runs whenever context fills, so it gets a hard cap up front. Past the cap we
# stop appending and say so in the notice — a truncated archive that admits it beats a
# full disk.
_MAX_OFFLOAD_BYTES = 32 * 1024 * 1024


def _offload_path() -> str:
    """Per-process offload file, resolved fresh each call. Never under cwd: the eval
    harness asserts plan-mode tasks leave the project tree untouched, and a stray file in
    the repo would fail that check (and pollute a user's git status). HOME is read live
    because the eval harness repoints it per task."""
    base = os.environ.get("CHAD_OFFLOAD_DIR") or os.path.join(
        os.path.expanduser("~"), ".chad", "offload")
    return os.path.join(base, f"compacted-{os.getpid()}.md")


def _spill(dropped) -> tuple[str, bool]:
    """Append the pre-compaction text of `dropped` messages to the offload file. Returns
    (path, capped). Best-effort: a failed spill must not abort the turn, since compaction
    itself is what stands between this step and an over-limit context."""
    path = _offload_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        capped = os.path.exists(path) and os.path.getsize(path) >= _MAX_OFFLOAD_BYTES
        if capped:
            return path, True
        stamp = time.strftime("%H:%M:%S")
        chunks = [f"\n\n## compaction at {stamp} — {len(dropped)} message(s)\n"]
        for role, name, content in dropped:
            who = f"{role}:{name}" if name else role
            chunks.append(f"\n### {who}\n\n{content}\n")
        with open(path, "a", errors="replace") as f:
            f.write("".join(chunks))
        return path, False
    except OSError as e:
        log.warning("compaction offload failed: %s", e)
        return path, False


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


# When compaction cannot get under ctx_limit (the protected floor — system prompt +
# skills + recent window + active query — exceeds the limit), re-running it every step
# is pure loss: each attempt mutates the transcript, which diverges the prefix and
# forces a FULL re-prefill on the non-trimmable cache (measured: 26-28s per step, 7
# steps consecutive, traces/session2.jsonl). The latch skips re-compaction until the
# transcript has actually grown this many tokens past the recorded floor; the engine's
# memory clamps (plan 075 WS1.1) make the small over-limit excursion safe.
_OVERLIMIT_REARM = 2048


def compact_if_needed(messages, render, emit, ctx_limit, prompt_ids, state=None):
    """Context compaction for long agentic sessions. On a non-trimmable cache
    (Ornith) any prefix change forces a full re-prefill, so compaction is
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
    # Drop the previous compaction's notice before doing anything else. It described the
    # last compaction, not this one, and left in place the notices would stack once per
    # window-fill. Re-appending a fresh one at the tail also keeps the guidance adjacent
    # to the newest tool result, which is the only place it reliably gets read.
    messages[:] = [m for m in messages if _NOTICE_TAG not in m.get("content", "")]
    # Pre-compaction text, held by reference so the passes below can be diffed against it.
    snapshot = [(m, m.get("role"), m.get("name"), m.get("content", "")) for m in messages]

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
    # NEVER delete the system prompt or the active user query: removing the query
    # makes the chat template raise "No user query found in messages" and crashes
    # the turn. The protected span is those two plus the last KEEP_RECENT messages
    # — NOT "everything after the last user query". That older rule made this pass
    # dead code in a sub-agent, whose only user message is the task prompt at index
    # 1: the deletable range collapsed to range(1, 1) and every sub-agent fell
    # straight through to pass 4, which shredded the very search results it had just
    # gathered. Bounding by a trailing window instead reclaims from old churn in
    # both agents while keeping the working set verbatim.
    def _last_user_idx():
        for i in range(len(messages) - 1, 0, -1):
            if messages[i].get("role") == "user":
                return i
        return None
    KEEP_RECENT = 6  # trailing messages held verbatim (the live working set)
    BATCH = 8  # delete up to this many oldest messages between budget re-checks
    guard = 0
    # Ablating this restores the old ceiling — everything after the last user query is
    # protected — which in a sub-agent (one user message, at index 1) collapses the
    # deletable range to range(1, 1) and makes this whole pass dead code.
    trailing_window = levers.enabled("subagent_compact_window")
    while cur_len() > target and guard < 500:
        lu = _last_user_idx()
        ceil = (max(1, len(messages) - KEEP_RECENT) if trailing_window
                else (lu if lu is not None else len(messages)))
        # Oldest deletable indices: after the system prompt, before the recent window,
        # never the active user query, and never an activated skill message (durable
        # guidance is kept verbatim). Collect up to BATCH of them in one scan so we
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
    # keep_recent=0 shredded it unread — the sub-agent would grep, lose the hits to
    # compaction, and grep again until it hit its step cap with nothing to return.
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

    # What did the passes actually destroy? Compare each pre-compaction message against
    # its post-compaction self by object identity (indices shift when pass 3 deletes, and
    # holding the dict keeps it alive so a freed id can't be reused under us). A message
    # counts as lost if it was deleted outright OR if its content changed — pass 1 strips
    # <think>, passes 2/4 clip head/tail, and both are content the model can no longer see.
    survivors = {id(m) for m in messages}
    dropped = [(role, name, content) for m, role, name, content in snapshot
               if id(m) not in survivors or m["content"] != content]

    path, capped = ("", False)
    if dropped and levers.enabled("compact_offload"):
        path, capped = _spill(dropped)
    noticed = False
    if dropped and levers.enabled("compact_notice"):
        where = ""
        if path and not capped:
            where = (f" The full untrimmed text of everything removed is saved at "
                     f"{path} — `read` that file if you need any of it.")
        elif capped:
            where = (" The offload archive hit its size cap, so the removed text is gone "
                     "for good; re-run the tool if you need it.")
        messages.append({"role": "tool", "name": "read", "content": (
            f"{_NOTICE_TAG} Your context was compacted to fit the window. "
            f"{len(dropped)} earlier message(s) were trimmed or dropped, including tool "
            f"output you had already read. Any text marked "
            f"\"{_COLLAPSED}\" is INCOMPLETE: do not answer from it, and do not assume "
            f"you know a file's contents from a clipped read of it.{where} "
            f"If you need earlier file contents or command output, re-read the file or "
            f"re-run the command — do not recall it from memory. Then continue the task.")})
        noticed = True
        # The notice costs tokens, and it is appended after the passes above have already
        # reclaimed down to the ceiling. On the pathological path — one fresh read bigger
        # than the whole target — that can push the render back over ctx_limit, and an
        # over-limit return leaves the caller re-prefilling a context that only grows.
        # The ceiling is the invariant; the notice is guidance. Guidance yields.
        if cur_len() > ctx_limit:
            messages.pop()
            noticed = False
            log.warning("COMPACT notice dropped: would exceed ctx_limit=%d", ctx_limit)

    new_ids = render()
    # Log, not just emit: emit() only reaches the TUI, so a compaction that failed to
    # converge (or one inside a sub-agent, whose display is suppressed) left no trace at
    # all. `over_limit` is the one to alert on — it means the next step re-prefills a
    # context that only grows.
    log.info("COMPACT %d→%d tokens | limit=%d target=%d msgs=%d dropped=%d notice=%s%s",
             before, len(new_ids), ctx_limit, target, len(messages), len(dropped), noticed,
             " OVER-LIMIT" if len(new_ids) > ctx_limit else "")
    if state is not None:
        if len(new_ids) > ctx_limit:
            # Couldn't get under the limit: the protected floor is bigger than the
            # window. Latch, so the next steps don't thrash the warm cache retrying.
            if not state.get("overlimit_floor"):
                log.warning("COMPACT floor-bound: latching until context grows past "
                            "%d (+%d)", len(new_ids), _OVERLIMIT_REARM)
            state["overlimit_floor"] = len(new_ids)
        else:
            state.pop("overlimit_floor", None)
    emit("info", f"  [compacted context: {before}→{len(new_ids)} tokens]")
    return new_ids
