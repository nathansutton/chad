"""Unit tests for the deterministic post-edit syntax gate (syntaxgate.py, plan 038).

The gate rides a warning along in the SAME tool result when a write/edit *introduces* a
syntax error — never blocking, never touching a valid edit or a pre-existing parse error.
Pure + fast: no model load. Run: `uv run python test_syntaxgate.py`
"""

import os
import tempfile

from chad import tools
from chad.tools import tool_edit, tool_write

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def _tmp(name, content):
    d = tempfile.mkdtemp(prefix="syntaxgate_")
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(content)
    return p


def test_python():
    # A write that produces invalid Python is flagged, with the offending line number.
    p = _tmp("a.py", "")
    res = tool_write(p, "def f(:\n    pass\n")
    check("py bad write warns", "warning" in res and "no longer parses" in res, res)
    check("py warning has line info", "at line 1" in res, res)

    # A valid write is silent.
    res = tool_write(p, "def f():\n    return 1\n")
    check("py good write silent", "warning" not in res, res)

    # An edit that breaks the file is flagged; the edit still applies (never rolled back).
    res = tool_edit(p, "def f():", "def f(:")
    check("py bad edit warns", "warning" in res, res)
    with open(p) as f:
        check("py bad edit still applied", "def f(:" in f.read())

    # A failed edit (target absent) leaves the file unchanged -> no warning.
    p = _tmp("b.py", "def g():\n    return 2\n")
    res = tool_edit(p, "not_present_anywhere", "x")
    check("py failed edit silent", "warning" not in res and "not found" in res, res)


def test_tree_sitter_delta():
    # A C file that ALREADY has parse errors: an unrelated valid edit must NOT warn
    # (we only flag errors the edit itself introduced, never pre-existing ones).
    p = _tmp("c1.c", "int main( {  // deliberately broken header\nint x = 1;\n")
    res = tool_edit(p, "int x = 1;", "int x = 2;")
    check("ts pre-existing error not flagged", "warning" not in res, res)

    # A clean C file that an edit breaks IS flagged.
    p = _tmp("c2.c", "int main(){ return 0; }\n")
    res = tool_edit(p, "return 0;", "return 0")  # drop the semicolon
    check("ts newly-introduced error warns", "warning" in res, res)


def test_opt_out():
    os.environ["CHAD_NO_SYNTAX_GATE"] = "1"
    try:
        p = _tmp("c.py", "")
        res = tool_write(p, "def broken(:\n")
        check("CHAD_NO_SYNTAX_GATE disables gate", "warning" not in res, res)
    finally:
        del os.environ["CHAD_NO_SYNTAX_GATE"]


def test_symbol_edit():
    # A symbol replacement that produces invalid Python is flagged in the same result.
    p = _tmp("mod.py", "def area(w, h):\n    return w * h\n")
    res = tools.symbols.service().replace_symbol("area", "def area(w, h):\n    return w *", path=p)
    check("symbol bad replace warns", "warning" in res, res)
    # And a valid symbol replacement is silent.
    p = _tmp("mod2.py", "def area(w, h):\n    return w * h\n")
    res = tools.symbols.service().replace_symbol("area", "def area(w, h):\n    return w * h * 2", path=p)
    check("symbol good replace silent", "warning" not in res, res)


if __name__ == "__main__":
    test_python()
    test_tree_sitter_delta()
    test_opt_out()
    test_symbol_edit()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
