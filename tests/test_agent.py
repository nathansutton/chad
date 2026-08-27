"""Characterization tests for agent.py's pure, model-free helpers — `expand_mentions`
(the @file/@dir mention expander) and `_has_open_tool_call` (mid-call truncation
detection). The run_turn loop needs a model and is out of scope here; its decision
predicates live in test_agent_guards.py. Importing `agent` is cheap — it loads no
model and (as of) no longer builds a system prompt at import time.

A bug in `expand_mentions` resolves an email (`foo@bar.com`) as a path, or pulls in a
file the user never referenced; a bug in `_has_open_tool_call` misreads a truncated
turn. These pin the current contract.

Run: `uv run python tests/test_agent.py`
"""

import os
import tempfile

import pytest

from chad.agent import (
    Agent,
    _has_open_tool_call,
    close_unclosed_think,
    expand_mentions,
    split_inline_reasoning,
)

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def test_expand_mentions_resolves_file(monkeypatch, tmp_path):
    monkeypatch.chdir(str(tmp_path))
    with open("hi.py", "w") as f:
        f.write("print('hi')\n")
    text, resolved = expand_mentions("look at @hi.py please")
    check("file mention resolved", resolved == ["hi.py"], resolved)
    check("augmented text has Attached block", "[Attached" in text, text)
    check("augmented text references the file", "@hi.py" in text, text)
    # the file's contents are pulled in (via tool_read) so no read round-trip is needed
    check("file contents attached", "print('hi')" in text, text)


def test_expand_mentions_rejects_email_and_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(str(tmp_path))
    original = "email me at foo@bar.com about @does_not_exist.py"
    text, resolved = expand_mentions(original)
    check("nothing resolves", resolved == [], resolved)
    check("text unchanged when nothing resolves", text == original, text)


def test_expand_mentions_resolves_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(str(tmp_path))
    os.mkdir("sub")
    with open(os.path.join("sub", "a.txt"), "w") as f:
        f.write("x")
    text, resolved = expand_mentions("see @sub for details")
    check("dir mention resolved", resolved == ["sub"], resolved)
    check("dir produces a listing", "directory listing" in text, text)
    check("dir listing names entry", "a.txt" in text, text)


def test_expand_mentions_dedupes(monkeypatch, tmp_path):
    monkeypatch.chdir(str(tmp_path))
    with open("dup.py", "w") as f:
        f.write("# dup\n")
    _, resolved = expand_mentions("compare @dup.py with @dup.py again")
    check("repeated mention resolves once", resolved == ["dup.py"], resolved)


def test_has_open_tool_call():
    # An opened-but-unclosed <tool_call> (the write-blew-the-cap signature) -> True.
    check("open tool_call -> True", _has_open_tool_call('<tool_call>{"name": "write"'))
    # A balanced block -> False.
    check("closed tool_call -> False", _has_open_tool_call("<tool_call>{}</tool_call>") is False)
    # Same for the <function=...> spelling.
    check("open function= -> True", _has_open_tool_call("<function=write>partial"))
    check("closed function= -> False",
          _has_open_tool_call("<function=write></function>") is False)
    # Plain prose with no tool markers -> False.
    check("plain prose -> False", _has_open_tool_call("here is the answer") is False)


def test_close_unclosed_think():
    # A turn truncated mid-think (template auto-opened <think>, model never closed it):
    # close it so the stored turn re-tokenizes into a prefix of the live KV cache.
    check("unclosed think -> closed",
          close_unclosed_think("reasoning cut off here", True) == "reasoning cut off here\n</think>")
    # Already-closed think (normal completed turn) -> untouched.
    closed = "reasoning</think>the answer"
    check("closed think untouched", close_unclosed_think(closed, True) == closed)
    # Thinking disabled (--no-think) -> never inject a tag.
    check("no-think untouched", close_unclosed_think("plain answer", False) == "plain answer")
    # A stray opening <think> the model wrote INSIDE its own reasoning does not close
    # anything. Only `</think>` decides. Reading the stray tag as "already handled" left
    # the turn unclosed, which is the full re-prefill this helper exists to avoid.
    check("stray <think>, no close -> closed",
          close_unclosed_think("thinking <think> wait", True) == "thinking <think> wait\n</think>")
    # ...and one that IS closed stays untouched, stray tag or not.
    stray_closed = "thinking <think> wait</think>answer"
    check("stray <think>, already closed -> untouched",
          close_unclosed_think(stray_closed, True) == stray_closed)
    # Empty text -> no spurious tag.
    check("empty untouched", close_unclosed_think("", True) == "")


def test_split_inline_reasoning():
    # A normal completed turn: reasoning up to </think>, action after it. chad stores the
    # stream verbatim (the generation prompt emitted the opening tag), so there is no
    # leading <think> to strip.
    m = {"role": "assistant", "content": "weighing it up\n</think>\n\ncall the tool"}
    out = split_inline_reasoning(m)
    check("reasoning lifted", out["reasoning_content"] == "weighing it up", out)
    check("content is the action", out["content"] == "call the tool", out)
    check("input not mutated", m["content"].startswith("weighing"), m)
    # An explicit opening <think> (some backends re-emit it) is dropped from the
    # reasoning, matching the recovery Ornith's template does.
    out = split_inline_reasoning({"role": "assistant", "content": "<think>\nwhy\n</think>\n\nans"})
    check("opening tag stripped", out["reasoning_content"] == "why", out)
    # No </think> at all: a --no-think / no-think-escalation turn, whose generation prompt
    # already carried the empty block. Untouched, so its render still matches the cache.
    nothink = {"role": "assistant", "content": "<tool_call>...</tool_call>"}
    check("no-think turn untouched", split_inline_reasoning(nothink) is nothink)
    # Non-assistant turns never carry reasoning.
    tool = {"role": "tool", "name": "bash", "content": "x\n</think>\ny"}
    check("tool turn untouched", split_inline_reasoning(tool) is tool)
    check("user turn untouched",
          split_inline_reasoning({"role": "user", "content": "a</think>b"})["content"] == "a</think>b")
    # A stray <think> mid-reasoning is reasoning text, not a delimiter. Splitting on the
    # LAST tag discarded everything before it — reasoning the model really emitted, gone
    # from the transcript, and the shortened re-render no longer extends the KV cache.
    out = split_inline_reasoning(
        {"role": "assistant", "content": "first half <think> second half\n</think>\n\nans"})
    check("stray tag keeps whole reasoning",
          out["reasoning_content"] == "first half <think> second half", out)
    check("stray tag keeps action", out["content"] == "ans", out)


def test_reasoning_split_probe_classifies_templates():
    """The probe must turn the split ON only for templates that need it. Ornith-class
    templates (which recover inline `</think>` themselves) must stay OFF so their
    rendered prompt is byte-identical to before this change; non-thinking templates
    (which would DROP the reasoning) must stay off too."""
    class FakeTok:
        def __init__(self, kind):
            self.kind = kind

        def apply_chat_template(self, msgs, add_generation_prompt=False,
                                enable_thinking=True, tokenize=False, **kw):
            out = []
            for m in msgs:
                if m["role"] != "assistant":
                    out.append(m["content"])
                    continue
                r, c = m.get("reasoning_content", ""), m["content"]
                if self.kind == "ornith" and not r and "</think>" in c:
                    r, _, c = c.partition("</think>")   # template recovers it itself
                if self.kind == "plain":
                    out.append(c)                       # ignores reasoning_content
                else:
                    out.append(f"<think>{r.strip()}</think>{c.strip()}")
            return "|".join(out)

    def probe(kind):
        a = Agent.__new__(Agent)
        a.engine = type("E", (), {"tok": FakeTok(kind)})()
        return a._reasoning_split_supported()

    check("qwen-class template -> split ON", probe("qwen") is True)
    check("ornith-class template -> split off", probe("ornith") is False)
    check("non-thinking template -> split off (would lose reasoning)", probe("plain") is False)

    # The verdict is resolved once and cached (it renders two probe transcripts).
    a = Agent.__new__(Agent)
    a.engine = type("E", (), {"tok": FakeTok("qwen")})()
    a._reasoning_split_supported()
    a.engine = type("E", (), {"tok": FakeTok("ornith")})()
    check("verdict cached", a._reasoning_split_supported() is True)

    # An exotic template that raises must fall back to OFF (pre-existing behavior),
    # never break the turn.
    class Boom:
        def apply_chat_template(self, *a, **k):
            raise ValueError("no user query found in messages")
    a = Agent.__new__(Agent)
    a.engine = type("E", (), {"tok": Boom()})()
    check("unprobeable template -> off", a._reasoning_split_supported() is False)


def test_stored_turn_extends_the_kv_cache():
    """The invariant the two helpers exist to hold: whatever the model generated, the
    NEXT render must begin with the bytes already in the KV cache.

    Break it and the server re-evaluates the whole turn it just produced — the longer
    the reasoning, the more it costs. Testing the property rather than each helper is
    what catches a turn shape neither of them was written for; the shapes below are all
    ones a real 27B emits. Model-free: `render` is a stand-in for Qwen3.8's template,
    which reads `reasoning_content` and does NOT recover inline `</think>` itself.
    """
    def render(msgs):
        out = []
        for m in msgs:
            role = m["role"]
            if role == "assistant":
                out.append("<|im_start|>assistant\n<think>\n"
                           + (m.get("reasoning_content") or "")
                           + "\n</think>\n\n" + (m.get("content") or "") + "<|im_end|>\n")
            else:
                out.append(f"<|im_start|>{role}\n{m.get('content') or ''}<|im_end|>\n")
        # The generation prompt auto-opens the think block, so the turn continues inside it.
        return "".join(out) + "<|im_start|>assistant\n<think>\n"

    prefix = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "task"}]
    reasoning = "weighing the ragged-row rule " * 8
    call = "<tool_call>\n<function=bash>\n</function>\n</tool_call>"

    shapes = {
        "normal turn": reasoning + "\n</think>\n\nrunning it." + call,
        "cut off mid-think": reasoning,
        "stray <think>, never closed": reasoning + "<think> wait " + reasoning,
        "stray <think>, then closed": (reasoning + "<think> wait " + reasoning
                                       + "\n</think>\n\nrunning it." + call),
    }
    for name, generated in shapes.items():
        # What the server holds after decoding this turn: the prompt it was given, then
        # the raw bytes it emitted.
        cached = render(prefix) + generated
        # What chad sends next, once the turn is stored and a tool result appended.
        stored = {"role": "assistant", "content": close_unclosed_think(generated, True)}
        nxt = render([split_inline_reasoning(m) for m in
                      prefix + [stored, {"role": "tool", "name": "bash", "content": "ok"}]])
        keep = len(os.path.commonprefix([cached, nxt]))
        check(f"{name}: render extends the cache",
              keep >= len(cached) - len("\n"), f"diverged after {keep} of {len(cached)} chars")
        # Reasoning the model emitted must survive into the transcript, all of it.
        check(f"{name}: reasoning kept",
              reasoning.strip() in split_inline_reasoning(stored).get("reasoning_content", ""))

if __name__ == "__main__":
    with pytest.MonkeyPatch.context() as mp:
        with tempfile.TemporaryDirectory() as d:
            test_expand_mentions_resolves_file(mp, d)
    with pytest.MonkeyPatch.context() as mp:
        with tempfile.TemporaryDirectory() as d:
            test_expand_mentions_rejects_email_and_missing(mp, d)
    with pytest.MonkeyPatch.context() as mp:
        with tempfile.TemporaryDirectory() as d:
            test_expand_mentions_resolves_dir(mp, d)
    with pytest.MonkeyPatch.context() as mp:
        with tempfile.TemporaryDirectory() as d:
            test_expand_mentions_dedupes(mp, d)
    test_has_open_tool_call()
    test_close_unclosed_think()
    test_split_inline_reasoning()
    test_reasoning_split_probe_classifies_templates()
    test_stored_turn_extends_the_kv_cache()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


# --- /ctx: where the window actually went -------------------------------------

def test_ctx_spans_splits_a_transcript():
    """`context N` says how full the window is and nothing about why. The split is
    what a user can act on — a 40k skill body, an eager MCP server's schemas and a
    transcript of think blocks all read identically on the bare gauge."""
    from chad.agent import _ctx_spans
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<think>" + "t" * 40 + "</think>answer"},
        {"role": "tool", "name": "bash", "content": "o" * 100},
        {"role": "assistant", "content": "no thinking here"},
        {"role": "tool", "name": "bash", "content": "r" * 30},
        {"role": "tool", "name": "bash", "content": "r" * 20},
    ]
    think, tools, recent = _ctx_spans(msgs, len)   # 1 char == 1 "token"
    check("think residue counts only <think> spans", think == 40, think)
    check("tool results count every tool message", tools == 150, tools)
    check("trailing run is the last step's results only", recent == 50, recent)

    # No trailing tool run (the transcript ends on an assistant turn) -> 0, not "all".
    _, _, none_recent = _ctx_spans(msgs[:-2], len)
    check("no trailing tool run reports 0", none_recent == 0, none_recent)
    # An unterminated <think> is not a span: half a block must not be priced as one.
    open_think, _, _ = _ctx_spans([{"role": "assistant", "content": "<think>abc"}], len)
    check("unterminated think block counts 0", open_think == 0, open_think)


def test_ctx_breakdown_adds_up():
    """The parts must reconcile to the total — `chat` is the remainder on purpose, so
    template scaffolding lands there instead of being attributed to content that did
    not produce it."""
    from chad.agent import Agent, format_ctx_breakdown
    from test_agent_e2e import ScriptedEngine
    agent = Agent(ScriptedEngine(["done"]), mode="yolo", thinking=False)
    agent.messages += [
        {"role": "user", "content": "do a thing"},
        {"role": "assistant", "content": "<think>" + "t" * 400 + "</think>ok"},
        {"role": "tool", "name": "bash", "content": "o" * 800},
    ]
    bd = agent.ctx_breakdown()
    check("system + schemas + history == total",
          bd["system"] + bd["schemas"] + bd["history"] == bd["total"], bd)
    check("history decomposes exactly",
          bd["think"] + bd["tools"] + bd["chat"] == bd["history"], bd)
    check("every part is non-negative", all(v >= 0 for v in bd.values()), bd)
    check("the limit shown is the compaction trigger", bd["limit"] == agent.ctx_limit, bd)
    lines = format_ctx_breakdown(bd)
    check("formats one line per part", len(lines) >= 7, lines)
    check("names the compaction trigger", f"{bd['limit']:,}" in lines[0], lines[0])
