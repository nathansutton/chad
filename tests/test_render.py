"""Tests for the display-only `_is_err` heuristic and `render_tool_result` styling
(src/chad/render.py).

`_is_err` decides whether a tool result is shown with error styling. It is a cheap
keyword scan, so the risk is false positives: legitimate output that merely *looks* like
an error. Plan 044 item 7 tightened it to require a leading `[` AND to scan only the
first line, so a `read`/grep of `[`-leading multi-line content (a JSON array, a
TOML/markdown doc) can't smuggle an error keyword in from line 2. These snapshot the
current contract: real chad diagnostics flag, ordinary bracket-leading content doesn't.

Run: `uv run python test_render.py`
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from chad.render import _is_err, render_tool_result  # noqa: E402

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


def test_render_read_of_bracket_content_is_not_error():
    # Snapshot: reading a file that starts with `[` and mentions 'error' downstream shows
    # a line count (muted), NOT the red error line.
    content = '[\n  {"status": "error"},\n  {"status": "ok"}\n]'
    events = _emits("read", {}, content)
    check("read snapshot is a line count", events == [("muted", "  ⎿ 4 lines")], repr(events))


def test_render_real_error_uses_error_style():
    # Snapshot: a genuine error result emits a single 'error' event with the first line.
    events = _emits("read", {}, "[no such file: /x/y.py]")
    check("error snapshot uses error kind",
          events == [("error", "  ⎿ no such file: /x/y.py")], repr(events))


if __name__ == "__main__":
    test_is_err_flags_real_diagnostics()
    test_is_err_ignores_legitimate_output()
    test_render_read_of_bracket_content_is_not_error()
    test_render_real_error_uses_error_style()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
