"""Tier-1 (no model) tests for the "Claude-Code feel" pack — the pure,
unit-testable slices of items 2/3/4. The prompt_toolkit Completer/menu wiring and the
scrollback-preserving pinned-region redraw are verified manually (see the plan); here we
pin the *logic*:

  * item 2 — slash-command matching, `@`-token extraction, and IGNORE_DIRS-aware path
    completion (the completer itself just forwards to these);
  * item 3 — FileHistory at ~/.chad/history is created 0600 only when opted in
    (CHAD_SESSION_LOG); off by default and under CHAD_NO_SESSION_LOG it stays in-memory;
  * item 4 — pygments highlighting is byte-identical to the plain path when the import is
    unavailable (monkeypatched off) and adds color when present, with the +/- diff
    coloring left as the outer layer.
"""
import os
import stat
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad import render  # noqa: E402
from chad.tui import (  # noqa: E402
    _make_history,
    at_path_token,
    path_matches,
    slash_matches,
)

# -- item 2: completion helpers ---------------------------------------------

def test_slash_matches_prefix_and_multiword():
    assert [c for c, _ in slash_matches("/mo")] == ["/model", "/mode"]
    # a multi-word command completes from its shared prefix
    assert set(c for c, _ in slash_matches("/mcp")) >= {"/mcp", "/mcp trust", "/mcp login"}
    assert [c for c, _ in slash_matches("/mcp ")] == ["/mcp trust", "/mcp login"]
    # non-slash / multiline input never offers command completions
    assert slash_matches("hello") == []
    assert slash_matches("/help\nmore") == []
    # once the text runs past a known command (typing an arg) nothing matches
    assert slash_matches("/mcp login myserver") == []


def test_at_path_token_extraction():
    assert at_path_token("read @src/ch") == "src/ch"
    assert at_path_token("@") == ""            # bare @ → complete cwd
    assert at_path_token("no token here") is None
    assert at_path_token("") is None
    # only the token under the cursor (last whitespace-delimited chunk) counts
    assert at_path_token("look at @a/b then foo") is None
    assert at_path_token("look at foo @a/b") == "a/b"


def test_path_matches_respects_ignore_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "keep.py").write_text("x")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / ".git").mkdir()            # in IGNORE_DIRS
    (tmp_path / "node_modules").mkdir()    # in IGNORE_DIRS
    got = path_matches("", cwd=str(tmp_path))
    assert "src/" in got and "keep.py" in got
    assert ".git/" not in got and "node_modules/" not in got
    assert ".hidden" not in got            # dotfiles hidden until a dot is typed
    # a partial narrows, and a leading dot opts back into dotfiles
    assert path_matches("ke", cwd=str(tmp_path)) == ["keep.py"]
    assert path_matches(".hid", cwd=str(tmp_path)) == [".hidden"]
    # nonexistent directory → no crash, empty list
    assert path_matches("nope/x", cwd=str(tmp_path)) == []


# -- item 3: persistent history ---------------------------------------------

def test_file_history_opt_in_is_created_0600(tmp_path, monkeypatch):
    # Persistent history is a local trace, so it follows the same opt-in gate as the
    # session log: only written when CHAD_SESSION_LOG is set.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CHAD_SESSION_LOG", "1")
    monkeypatch.delenv("CHAD_NO_SESSION_LOG", raising=False)
    h = _make_history()
    p = tmp_path / ".chad" / "history"
    assert type(h).__name__ == "FileHistory"
    assert p.exists()
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


def test_history_off_by_default_is_in_memory(tmp_path, monkeypatch):
    # Privacy-first default: with no opt-in, history stays in memory and ~/.chad/history
    # is never written.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("CHAD_SESSION_LOG", raising=False)
    monkeypatch.delenv("CHAD_NO_SESSION_LOG", raising=False)
    h = _make_history()
    assert type(h).__name__ == "InMemoryHistory"
    assert not (tmp_path / ".chad" / "history").exists()


def test_history_no_session_log_overrides_opt_in(tmp_path, monkeypatch):
    # The hard kill switch wins even when the opt-in is set.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("CHAD_SESSION_LOG", "1")
    monkeypatch.setenv("CHAD_NO_SESSION_LOG", "1")
    h = _make_history()
    assert type(h).__name__ == "InMemoryHistory"
    assert not (tmp_path / ".chad" / "history").exists()


# -- item 4: optional pygments highlighting ---------------------------------

def _diff_events(**kw):
    out = []
    render._emit_diff(lambda k, t: out.append((k, t)), "a = 1", "a = 2",
                      filename="x.py", **kw)
    return out


def test_highlight_plain_path_byte_identical_without_pygments(monkeypatch):
    # Monkeypatch the import away: the emitted diff must be byte-for-byte the pre-042
    # output (no ANSI beyond what the +/- emit kinds add downstream).
    monkeypatch.setattr(render, "_HAS_PYGMENTS", False)
    assert _diff_events() == [
        ("muted", "  ⎿ +1 -1"),
        ("del", "  - a = 1"),
        ("add", "  + a = 2"),
    ]
    # the helper itself is an identity function when disabled
    assert render._highlight_code("def f(): pass", "x.py") == "def f(): pass"


def test_highlight_adds_color_when_pygments_present(monkeypatch):
    import pytest
    pytest.importorskip("pygments")            # optional extra; skip on a bare install
    monkeypatch.setattr(render, "_HAS_PYGMENTS", True)
    hl = render._highlight_code("def f(): pass", "x.py")
    assert "\033[" in hl                       # pygments injected ANSI token colors
    assert "f(): pass" in _strip_ansi(hl)      # ...without dropping the code content


def _strip_ansi(s):
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)
