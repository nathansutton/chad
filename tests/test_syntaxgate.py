"""Unit tests for the deterministic post-mutation syntax warning (syntaxgate.py).

The gate rides a warning along in the SAME tool result when a write/edit *introduces* a
syntax error — never blocking, never touching a valid edit or a pre-existing parse error.
Pure + fast: no model load. Run: `uv run python tests/test_syntaxgate.py`
"""

import os
import tempfile

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
    # A write that lands invalid Python carries the warning in the SAME result.
    p = _tmp("a.py", "")
    res = tool_write(p, "def f(:\n    pass\n")
    check("py bad write lands with warning", res.startswith("[wrote"), res)
    check("py write warning has line info", "at line 1" in res, res)

    # A valid write is silent.
    res = tool_write(p, "def f():\n    return 1\n")
    check("py good write silent", "warning" not in res, res)

    # An ALREADY-broken file stays overwritable — the whole-file rewrite is the
    # repair path and must never be stranded; it still warns while broken.
    p = _tmp("w.py", "def broken(:\n")
    res = tool_write(p, "def broken(:\n    x = 1\n")
    check("py broken->broken write lands", res.startswith("[wrote"), res)
    check("py broken->broken write still warns", "no longer parses" in res, res)

    # An edit that breaks the file lands, with the warning in the result.
    p = _tmp("a2.py", "def f():\n    return 1\n")
    res = tool_edit(p, "def f():", "def f(:")
    check("py bad edit warns", "no longer parses" in res, res)

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

    # A clean C file that an edit breaks gets the warning in the result.
    before = "int main(){ return 0; }\n"
    p = _tmp("c2.c", before)
    res = tool_edit(p, "return 0;", "return 0")  # drop the semicolon
    check("ts newly-introduced error warns", "syntax error" in res, res)

    # A brand-NEW ts-lang file with broken content lands and warns too.
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


def test_plain_text_never_policed():
    # The language pack maps .txt to VIMDOC, so plain-text deliverable writes
    # (answer.txt / secret.txt / requirements.txt — the benchmark README finding)
    # were grammar-checked and warned on exactly the
    # deliverable-landing write. Prose/data formats are now excluded from every gate.
    for name, content in (
        ("answer.txt", "The flag is: ABC-123\nsecond line < > { weird ] chars\n"),
        ("requirements.txt", "numpy>=1.24\nscipy==1.11.*\n"),
        ("notes.md", "# heading\n<unclosed <tag [bracket\n"),
        ("data.csv", 'a,b\n1,"unclosed quote\n'),
    ):
        p = _tmp(name, "")
        res = tool_write(p, content)
        check(f"{name} write lands", res.startswith("[wrote"), res)
        check(f"{name} write has no syntax warning",
              "warning" not in res and "rejected" not in res, res)
        res = tool_edit(p, content.splitlines()[0], "replaced first line")
        check(f"{name} edit lands without warning",
              "warning" not in res and "rejected" not in res, res)
    # Sanity: real code languages are still policed (guard against an over-broad list).
    p = _tmp("still.py", "")
    res = tool_write(p, "def f(:\n")
    check("python still policed", "no longer parses" in res, res)


if __name__ == "__main__":
    test_python()
    test_plain_text_never_policed()
    test_tree_sitter_delta()
    test_opt_out()
    test_symbol_edit()
    test_indent_reject_names_enclosing_symbol()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
