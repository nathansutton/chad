"""Tests for the compaction in-band notice + offload (levers: compact_notice/offload).

Compaction used to be invisible to the model: the passes shrink the transcript, the TUI
prints a line, and the next turn sees a head/tail-clipped tool result with nothing saying
so. The model then answers from the clipped text as though it had read the whole thing.

Two properties matter and are asserted here:

  1. The notice lands at the TAIL, next to the data the model is about to read. Placement
     is the mechanism — the same words in a tool *description* demonstrably do nothing;
     they work in what the tool *returns*.
  2. Staying under ctx_limit outranks the notice. The notice costs tokens and is appended
     after the passes have already reclaimed to the ceiling, so on the pathological path
     (one fresh read larger than the whole target) it must be dropped rather than push the
     render back over the limit — an over-limit return leaves the caller re-prefilling a
     context that only grows.

Also asserted: notices do not stack across successive compactions, and the offload file
never lands inside cwd (the eval harness fails any plan-mode task that dirties the repo).
"""
import os

from chad import levers
from chad.compaction import _COLLAPSED, _NOTICE_TAG, compact_if_needed


def make_render(messages):
    def render():
        return list("".join(m["content"] for m in messages))
    return render


def noop_emit(*_a, **_k):
    pass


def _transcript(n_tools=6, hits=140):
    msgs = [{"role": "system", "content": "S" * 3000},
            {"role": "user", "content": "find the retry handling"}]
    for i in range(n_tools):
        msgs.append({"role": "assistant", "content": f"<think>r{i}</think>grep"})
        msgs.append({"role": "tool", "name": "grep", "content": "\n".join(
            f"src/m{i}/f{j}.py:{j}: def handle_{j}() " + "x" * 55 for j in range(hits))})
    return msgs


def _notices(msgs):
    return [m for m in msgs if _NOTICE_TAG in m.get("content", "")]


def test_notice_is_appended_at_the_tail(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    msgs = _transcript()
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=len(render()) // 2,
                      prompt_ids=render())
    assert len(_notices(msgs)) == 1
    assert _NOTICE_TAG in msgs[-1]["content"], "notice must sit next to the newest data"
    body = msgs[-1]["content"]
    assert "re-read" in body or "re-run" in body
    assert "do not" in body.lower()


def test_notice_names_the_offload_path_and_the_file_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    msgs = _transcript()
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=len(render()) // 2,
                      prompt_ids=render())
    path = os.path.join(str(tmp_path), f"compacted-{os.getpid()}.md")
    assert path in msgs[-1]["content"], "the model must be told where the text went"
    assert os.path.exists(path)
    archived = open(path).read()
    assert "handle_0()" in archived, "offload must hold the UNtrimmed original text"


def test_offload_never_lands_in_cwd(monkeypatch, tmp_path):
    """A stray file in the repo fails the plan-mode evals' 'project untouched' check."""
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.delenv("CHAD_OFFLOAD_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    from chad.compaction import _offload_path
    assert not _offload_path().startswith(os.getcwd() + os.sep)
    assert str(tmp_path) in _offload_path()


def test_notices_do_not_stack_across_compactions(monkeypatch, tmp_path):
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    msgs = _transcript(n_tools=3)
    render = make_render(msgs)
    ctx = 16000 * 4
    for step in range(6):
        msgs.append({"role": "assistant", "content": f"<think>t{step}</think>grep"})
        msgs.append({"role": "tool", "name": "grep", "content": "\n".join(
            f"src/s{step}/f{j}.py: hit {j} " + "y" * 55 for j in range(140))})
        compact_if_needed(msgs, render, noop_emit, ctx_limit=ctx, prompt_ids=render())
        assert len(_notices(msgs)) <= 1, "a stale notice must be stripped, not stacked"


def _oversized():
    """One fresh read larger than any target: compaction can only clamp it head/tail, so
    the converged size is a floor no ctx_limit can push below."""
    return [{"role": "system", "content": "S" * 200},
            {"role": "user", "content": "read the file"},
            {"role": "tool", "name": "read",
             "content": "\n".join(f"{i}: line " + "y" * 70 for i in range(4000))}]


def test_ceiling_outranks_the_notice(monkeypatch, tmp_path):
    """Converging under ctx_limit is mandatory; the notice is a heuristic. When both can't
    hold, the notice is what yields. Measured rather than guessed: compaction has a floor
    (system + user + the head/tail-clamped read) that no ctx_limit can push below, so we
    first learn that floor with the notice off, then re-run with a ceiling that clears the
    floor but cannot also fit the notice."""
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("CHAD_DISABLE", "compact_notice")
    msgs = _oversized()
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=1, prompt_ids=render())
    floor = len(render())

    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    msgs = _oversized()
    render = make_render(msgs)
    ctx = floor + 50  # room to converge, nowhere near room for the notice
    compact_if_needed(msgs, render, noop_emit, ctx_limit=ctx, prompt_ids=render())
    assert not _notices(msgs), "notice must be dropped rather than blow the ceiling"
    assert len(render()) <= ctx, "compaction must never return over the hard limit"


def test_notice_survives_when_there_is_room_for_it(monkeypatch, tmp_path):
    """The converse of the above: yielding is for the pathological path only. Given an
    ordinary transcript with headroom, the notice must actually be delivered."""
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    msgs = _transcript()
    render = make_render(msgs)
    ctx = len(render()) // 2
    compact_if_needed(msgs, render, noop_emit, ctx_limit=ctx, prompt_ids=render())
    assert len(_notices(msgs)) == 1
    assert len(render()) <= ctx


def test_levers_switch_each_half_off_independently(monkeypatch, tmp_path):
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    path = os.path.join(str(tmp_path), f"compacted-{os.getpid()}.md")

    monkeypatch.setenv("CHAD_DISABLE", "compact_notice")
    msgs = _transcript()
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=len(render()) // 2,
                      prompt_ids=render())
    assert not _notices(msgs), "compact_notice=off must inject nothing"
    assert os.path.exists(path), "offload is a separate lever and stays on"
    assert _COLLAPSED in "".join(m["content"] for m in msgs), "compaction still ran"

    os.remove(path)
    monkeypatch.setenv("CHAD_DISABLE", "compact_offload")
    msgs = _transcript()
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=len(render()) // 2,
                      prompt_ids=render())
    assert len(_notices(msgs)) == 1, "notice is a separate lever and stays on"
    assert not os.path.exists(path), "compact_offload=off must write nothing"
    assert str(tmp_path) not in msgs[-1]["content"], "no path to advertise"

    monkeypatch.setenv("CHAD_DISABLE", "compact_notice,compact_offload")
    msgs = _transcript()
    render = make_render(msgs)
    n_before = len(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=len(render()) // 2,
                      prompt_ids=render())
    assert not _notices(msgs) and not os.path.exists(path)
    assert len(msgs) < n_before, "with both levers off, compaction is exactly as it was"


def test_offload_respects_its_byte_cap(monkeypatch, tmp_path):
    """chad has already filled a disk once with an unbounded cache dir. Not twice."""
    monkeypatch.delenv("CHAD_DISABLE", raising=False)
    monkeypatch.setenv("CHAD_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr("chad.compaction._MAX_OFFLOAD_BYTES", 64)
    path = os.path.join(str(tmp_path), f"compacted-{os.getpid()}.md")
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x" * 128)  # already over the cap
    msgs = _transcript()
    render = make_render(msgs)
    compact_if_needed(msgs, render, noop_emit, ctx_limit=len(render()) // 2,
                      prompt_ids=render())
    assert os.path.getsize(path) == 128, "capped archive must not grow"
    assert "size cap" in msgs[-1]["content"], "and the model must be told it is gone"
    assert levers.enabled("compact_offload")
