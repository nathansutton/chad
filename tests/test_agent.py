"""Characterization tests for agent.py's pure, model-free helpers — `expand_mentions`
(the @file/@dir mention expander) and `_has_open_tool_call` (mid-call truncation
detection). The run_turn loop needs a model and is out of scope here; its decision
predicates live in test_agent_guards.py. Importing `agent` is cheap — it builds
SYSTEM_PROMPT at import but loads no model.

A bug in `expand_mentions` resolves an email (`foo@bar.com`) as a path, or pulls in a
file the user never referenced; a bug in `_has_open_tool_call` misreads a truncated
turn. These pin the current contract.

Run: `uv run python test_agent.py`
"""

import os
import tempfile

import pytest

from chad.agent import _has_open_tool_call, close_unclosed_think, expand_mentions

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
    # An explicit opening <think> in the text -> leave it (conservative; don't double-handle).
    check("explicit <think> untouched",
          close_unclosed_think("<think>partial", True) == "<think>partial")
    # Empty text -> no spurious tag.
    check("empty untouched", close_unclosed_think("", True) == "")


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
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
