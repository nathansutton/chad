"""Tests for the display-only `_is_err` heuristic and `render_tool_result` styling
(src/chad/render.py).

`_is_err` decides whether a tool result is shown with error styling. It is a cheap
keyword scan, so the risk is false positives: legitimate output that merely *looks* like
an error. It was tightened to require a leading `[` AND to scan only the
first line, so a `read`/grep of `[`-leading multi-line content (a JSON array, a
TOML/markdown doc) can't smuggle an error keyword in from line 2. These snapshot the
current contract: real chad diagnostics flag, ordinary bracket-leading content doesn't.

Run: `uv run python tests/test_render.py`
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad.render import (  # noqa: E402
    _is_err,
    _tilde,
    ansi_fragment,
    banner,
    piped_emit,
    render_passthrough,
    render_tool_result,
    strip_ansi,
)
from chad.tui import TUI  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def _emits(name, args, result):
    """Capture the (kind, text) events render_tool_result emits for one result."""
    out = []
    render_tool_result(lambda k, t: out.append((k, t)), name, args, result)
    return out


def test_ansi_fragment_parity():
    # The REPL emitter and the TUI must build byte-identical fragments for
    # every SHARED kind. `_ansi_for` routes through `ansi_fragment`, so pin that the two
    # paths can never silently diverge on a shared kind again. (stream/user are the two
    # deliberate divergences and are excluded here.)
    tui = object.__new__(TUI)  # shared kinds touch no instance state
    for kind in ("think", "tool", "add", "del", "error", "info", "muted"):
        frag = ansi_fragment(kind, "sample")
        assert frag is not None, kind
        assert frag == tui._ansi_for(kind, "sample"), kind
    # Non-shared / dropped kinds return None from the shared function.
    for kind in ("stream", "user", "stat", "ctx", "definitely-unknown"):
        assert ansi_fragment(kind, "x") is None, kind


def test_banner_states_the_session():
    # The startup banner names the product, model, context, mode, and cwd on three
    # rows beside the moai art — so a fresh session says what it is at a glance.
    out = banner("Ornith-9B", 75000, mode="plan", version="0.1.0",
                 cwd="/tmp/proj")
    rows = out.split("\n")
    assert len(rows) == 3
    assert "chad" in rows[0] and "v0.1.0" in rows[0]
    assert "Ornith-9B" in rows[1] and "75k context" in rows[1] and "plan mode" in rows[1]
    assert "/tmp/proj" in rows[2]
    # Art column is present on every row and equal width (left-justified).
    assert all("▙" in rows[0] or "▛" in r or "▝" in rows[2] for r in rows)


def test_banner_handles_unknown_context():
    # ctx_limit not yet resolved (None) must not crash the header.
    assert "context tbd" in banner("m", None)


def test_banner_advertises_the_window_the_box_gives():
    # The headline number is what the session ACTUALLY gets. When the RAM governor is
    # what bound it, both numbers are named — "84k" alone reads as the wrong model
    # having loaded, "262k" alone is context the run can never spend.
    limited = banner("Qwen3.8-27B", 84_300, native_ctx=262_144, version="0.1.0")
    assert "84k of 262k context" in limited
    # Governor not binding (big-RAM box): no second number to explain, so don't add one.
    roomy = banner("Qwen3.8-27B", 260_096, native_ctx=262_144, version="0.1.0")
    assert "260k context" in roomy and " of " not in roomy.split("\n")[1]
    # No native window known (config unreadable) -> the plain form, never a crash.
    assert "84k context" in banner("m", 84_300)


def _stub_tui(finalize):
    # A TUI shell exercising just the background-load handoff, without a real engine/app.
    import threading
    from types import SimpleNamespace
    tui = object.__new__(TUI)
    tui._model_ready = threading.Event()
    tui._wake = threading.Event()
    tui._load_error = None
    tui._describe_load_error = None
    tui.ctx_limit = 8192  # provisional, set pre-load
    tui.agent = SimpleNamespace(ctx_limit=8192)
    tui.engine = SimpleNamespace(effective_ctx=262144)
    tui.native_ctx = 262144
    tui._emit = lambda kind, text: None
    tui._finalize = finalize
    return tui


def test_background_load_adopts_ram_aware_ctx_limit():
    # The worker gate stays closed until the weights are in; then the provisional limit
    # is replaced by the RAM-aware one on both the TUI and the agent, and the worker wakes.
    tui = _stub_tui(lambda: (5.4, 175802))
    assert not tui._model_ready.is_set()
    tui._load_model()
    assert tui._model_ready.is_set()
    assert tui._wake.is_set()
    assert tui.ctx_limit == 175802 and tui.agent.ctx_limit == 175802
    assert tui._load_error is None


def test_background_load_failure_unblocks_worker():
    # A load crash must record the error AND open the gate, so a queued turn reports the
    # failure instead of hanging the session forever.
    def boom():
        raise RuntimeError("no metal")
    tui = _stub_tui(boom)
    tui._load_model()
    assert tui._model_ready.is_set()
    assert "no metal" in tui._load_error


def test_tilde_collapses_home():
    home = os.path.expanduser("~")
    assert _tilde(home) == "~"
    assert _tilde(os.path.join(home, "assist")) == os.path.join("~", "assist")
    assert _tilde("/etc/hosts") == "/etc/hosts"  # non-home paths untouched


def test_strip_ansi_drops_colour_codes():
    assert strip_ansi("\033[2mx\033[0m") == "x"


def test_piped_emit_keeps_stdout_for_the_answer(capsys):
    piped_emit("info", "hello")
    out = capsys.readouterr()
    assert out.err == "hello\n" and out.out == ""

    piped_emit("stream", "abc")
    assert capsys.readouterr().err == "abc"

    piped_emit("gen", "5")  # a live gauge: the TUI's, never scrollback
    out = capsys.readouterr()
    assert out.err == "" and out.out == ""


def test_is_err_flags_real_diagnostics():
    # Genuine chad error messages: leading `[`, keyword on the (single) first line.
    for r in ("[no such file: /x/y.py]",
              "[old string not found; no change made.]",
              "[old string appears 3 times; make it unique]",
              "[bad regex: unbalanced parenthesis]",
              "[timed out after 30s]",
              "[exit 1]\nTraceback (most recent call last):",
              "[unknown tool 'frobnicate'. Available: read, write]",
              "[no-op edit: old and new are identical; change the content or stop]"):
        check(f"flags {r[:24]!r}", _is_err(r) is True, r)


def test_is_err_ignores_legitimate_output():
    # Bracket-leading MULTI-LINE content whose keyword lives on a later line: a JSON
    # array read, a TOML doc read. Pre-044 these mis-styled as errors.
    json_arr = '[\n  {"level": "error", "msg": "boot"},\n  {"level": "info"}\n]'
    check("json array w/ 'error' on line 2 not an error", _is_err(json_arr) is False, json_arr)
    toml = "[tool.pytest.ini_options]\naddopts = \"--strict\"\n# no matches expected"
    check("toml section read not an error", _is_err(toml) is False, toml)
    # Non-bracket content is never an error regardless of keywords.
    check("plain text w/ 'error' not flagged", _is_err("compilation error on line 4") is False)
    # A bracket-leading result with NO keyword on the first line is not an error.
    check("bracketed non-error not flagged", _is_err("[replaced foo (3 lines)]") is False)


def test_render_real_error_uses_error_style():
    # Snapshot: a genuine error result emits a single 'error' event with the first line.
    events = _emits("read", {}, "[no such file: /x/y.py]")
    check("error snapshot uses error kind",
          events == [("error", "  ⎿ no such file: /x/y.py")], repr(events))



def _passthrough(result):
    """Capture the (kind, text) events render_passthrough emits for one `!command`."""
    shown = []
    render_passthrough(lambda kind, text: shown.append((kind, text)), result)
    return shown


def test_passthrough_empty_output():
    assert _passthrough("") == [("muted", "  ⎿ (no output)")]


def test_passthrough_short_output_is_shown_whole():
    shown = _passthrough("\n".join(f"line {i}" for i in range(1, 11)))
    assert len(shown) == 10
    assert all(kind == "muted" for kind, _ in shown)
    assert not any("not shown" in text for _, text in shown)


def test_passthrough_long_output_keeps_head_and_tail():
    shown = _passthrough("\n".join(f"line {i}" for i in range(1, 501)))
    texts = [text.strip().removeprefix("⎿ ") for _, text in shown]
    assert len(shown) == 201
    for kept in ("line 1", "line 100", "line 401", "line 500"):
        assert kept in texts
    assert "line 250" not in texts
    assert any("300 lines not shown" in t for t in texts)


def test_passthrough_failure_shows_its_output():
    shown = _passthrough("[exit 1]\nFAILED test_x\nassert 1 == 2")
    assert shown[0][0] == "error" and "exit 1" in shown[0][1]
    assert shown[1:] == [("muted", "     FAILED test_x"), ("muted", "     assert 1 == 2")]


def test_passthrough_timeout_shows_partial_output():
    shown = _passthrough("[timed out after 120s]\npartial")
    assert shown[0][0] == "error"
    assert any("partial" in text for _, text in shown[1:])


def test_passthrough_bracketed_output_is_not_an_error():
    shown = _passthrough("[1, 2, 3]\nnext")
    assert all(kind == "muted" for kind, _ in shown)
    assert [text.strip().removeprefix("⎿ ") for _, text in shown] == ["[1, 2, 3]", "next"]


def test_model_bash_result_is_still_six_lines():
    # The model's own tool calls stay terse: the user is watching an agent work.
    events = _emits("bash", {"command": "x"}, "\n".join(str(i) for i in range(20)))
    assert len(events) == 7
    assert events[-1] == ("muted", "     … +14 lines")


if __name__ == "__main__":
    test_is_err_flags_real_diagnostics()
    test_is_err_ignores_legitimate_output()
    test_render_real_error_uses_error_style()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
