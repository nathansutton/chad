"""Unit tests for the deterministic post-edit syntax gate (syntaxgate.py, plan 038).

The gate rides a warning along in the SAME tool result when a write/edit *introduces* a
syntax error — never blocking, never touching a valid edit or a pre-existing parse error.
Pure + fast: no model load. Run: `uv run python test_syntaxgate.py`
"""

import os
import tempfile

from chad import syntaxgate, tools
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
    # A write that would land invalid Python on a clean file is REJECTED with the file
    # untouched (plan 079) — write was the warn-only escape hatch that delivered 51 of
    # the 55 landed syntax breaks in the benchmark trace sweep.
    p = _tmp("a.py", "")
    res = tool_write(p, "def f(:\n    pass\n")
    check("py bad write rejected", res.startswith("[write rejected"), res)
    check("py write reject has line info", "at line 1" in res, res)
    with open(p) as f:
        check("py bad write did not land", f.read() == "")

    # A valid write is silent.
    res = tool_write(p, "def f():\n    return 1\n")
    check("py good write silent", "warning" not in res and "rejected" not in res, res)

    # An ALREADY-broken file stays overwritable with still-broken content — the
    # whole-file rewrite is the sanctioned repair path and must never be stranded.
    p = _tmp("w.py", "def broken(:\n")
    res = tool_write(p, "def broken(:\n    x = 1\n")
    check("py broken->broken write lands", res.startswith("[wrote"), res)
    check("py broken->broken write still warns", "no longer parses" in res, res)

    # An edit that breaks the file is REJECTED and reverted (plan 073): a landed break
    # is the precondition for the repair-of-garbage loop a small model can't win.
    p = _tmp("a2.py", "def f():\n    return 1\n")
    res = tool_edit(p, "def f():", "def f(:")
    check("py bad edit rejected", res.startswith("[edit rejected"), res)
    with open(p) as f:
        check("py bad edit reverted", "def f(:" not in f.read())

    # A failed edit (target absent) leaves the file unchanged -> no warning.
    p = _tmp("b.py", "def g():\n    return 2\n")
    res = tool_edit(p, "not_present_anywhere", "x")
    check("py failed edit silent", "warning" not in res and "not found" in res, res)


def test_tree_sitter_delta():
    # A C file that ALREADY has parse errors: an unrelated valid edit must NOT be
    # flagged or blocked (we only act on errors the edit itself introduced).
    p = _tmp("c1.c", "int main( {  // deliberately broken header\nint x = 1;\n")
    res = tool_edit(p, "int x = 1;", "int x = 2;")
    check("ts pre-existing error not flagged", "warning" not in res, res)
    check("ts pre-existing error still editable", res.startswith("[edited"), res)

    # A clean C file that an edit would break is REJECTED and reverted (plan 079: the
    # measured vm.js/ars.R class — non-Python breaks used to land with a warning and
    # compound through follow-up edits to reward-zero tasks).
    before = "int main(){ return 0; }\n"
    p = _tmp("c2.c", before)
    res = tool_edit(p, "return 0;", "return 0")  # drop the semicolon
    check("ts newly-introduced error rejected", res.startswith("[edit rejected"), res)
    check("ts reject says unparseable", "unparseable" in res, res)
    with open(p) as f:
        check("ts bad edit reverted", f.read() == before)

    # A whole-file write that would take a clean ts-lang file to broken is rejected too.
    p = _tmp("c3.c", before)
    res = tool_write(p, "int main( { return 0; }\n")
    check("ts clean->broken write rejected", res.startswith("[write rejected"), res)
    with open(p) as f:
        check("ts bad write did not land", f.read() == before)

    # But a brand-NEW ts-lang file is never rejected, only warned — a grammar quirk on
    # valid code must not block file creation (Python is held to ast.parse exactly;
    # tree-sitter grammars are not that trustworthy on content they've never seen).
    d = tempfile.mkdtemp(prefix="syntaxgate_")
    p = os.path.join(d, "new.c")
    res = tool_write(p, "int main( {\n")
    check("ts new-file broken write lands", res.startswith("[wrote"), res)
    check("ts new-file broken write warns", "warning" in res, res)


def test_opt_out():
    os.environ["CHAD_NO_SYNTAX_GATE"] = "1"
    try:
        p = _tmp("c.py", "")
        res = tool_write(p, "def broken(:\n")
        check("CHAD_NO_SYNTAX_GATE disables gate",
              "warning" not in res and "rejected" not in res, res)
    finally:
        del os.environ["CHAD_NO_SYNTAX_GATE"]


def test_symbol_edit():
    # A symbol replacement that produces invalid Python is REJECTED and reverted (plan
    # 073), and the message says the error is in the model's own code.
    p = _tmp("mod.py", "def area(w, h):\n    return w * h\n")
    res = tools.symbols.service().replace_symbol("area", "def area(w, h):\n    return w *", path=p)
    check("symbol bad replace rejected", res.startswith("[edit rejected"), res)
    check("symbol reject blames the sent code", "code you sent" in res, res)
    with open(p) as f:
        check("symbol bad replace reverted", "return w *\n" not in f.read())
    # And a valid symbol replacement is silent.
    p = _tmp("mod2.py", "def area(w, h):\n    return w * h\n")
    res = tools.symbols.service().replace_symbol("area", "def area(w, h):\n    return w * h * 2", path=p)
    check("symbol good replace silent", "warning" not in res and "rejected" not in res, res)


def test_indent_reject_names_enclosing_symbol():
    """B: when an in-function edit is reverted for breaking indentation, the message names
    the enclosing `Class/method` and points at replace_symbol — the stable path — so the
    model stops re-hand-indenting."""
    before = ("class Engine:\n"
              "    def _prefill(self, ids):\n"
              "        n = len(ids)\n"
              "        return n\n")
    # break indentation inside _prefill (over-indent the `n = len(ids)` line)
    after = before.replace("        n = len(ids)", "            n = len(ids)")
    p = _tmp("e.py", before)
    msg = syntaxgate.indent_reject(p, before, after)
    check("indent reject fired", msg and msg.startswith("[edit rejected"), msg)
    check("names enclosing method", "Engine/_prefill" in msg, msg)
    check("steers to replace_symbol", "replace_symbol" in msg, msg)

    # A module-level break has no enclosing function → no replace_symbol steer, but still
    # points at the line tools.
    before2 = "x = 1\ny = 2\n"
    after2 = "x = 1\n    y = 2\n"     # unexpected indent at module level
    p2 = _tmp("m.py", before2)
    msg2 = syntaxgate.indent_reject(p2, before2, after2)
    check("module-level reject fired", msg2 and msg2.startswith("[edit rejected"), msg2)
    check("module-level: no enclosing symbol steer", "You're editing inside" not in msg2, msg2)


if __name__ == "__main__":
    test_python()
    test_tree_sitter_delta()
    test_opt_out()
    test_symbol_edit()
    test_indent_reject_names_enclosing_symbol()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
