"""Unit tests for the deterministic post-mutation syntax warning (syntaxgate.py).

The gate rides a warning along in the SAME tool result when a write/edit *introduces* a
syntax error — never blocking, never touching a valid edit or a pre-existing parse error.
Pure + fast: no model load. Run: `uv run python tests/test_syntaxgate.py`
"""

import os
import tempfile

from chad.syntaxgate import _code_lang
from chad.tools import tool_edit, tool_write

_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "syntaxgate_corpus")

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


def test_cpp_header_judged_by_the_grammar_that_fits():
    # The language pack calls every `.h` C. Exercism's C++ track, like most C++, declares
    # namespaces and classes in `.h`, which the C grammar rejects: 18 of the 19 syntax
    # warnings in one 90-trial eval run were correct C++ headers. The header is judged by
    # whichever grammar fits it, and a real break under that grammar still warns.
    d = tempfile.mkdtemp(prefix="syntaxgate_")
    p = os.path.join(d, "yacht.h")
    res = tool_write(p, "#pragma once\n#include <string>\nnamespace yacht {\n"
                        "class score { public: int value() const; };\n}\n")
    check("cpp header write silent", "warning" not in res, res)
    res = tool_edit(p, "int value() const;", "int value() const")
    check("cpp header broken edit warns", "syntax error" in res, res)
    p = os.path.join(d, "plain.h")
    res = tool_write(p, "#ifndef PLAIN_H\n#define PLAIN_H\nint add(int a, int b);\n#endif\n")
    check("c header write silent", "warning" not in res, res)
    res = tool_edit(p, "int b);", "int b")
    check("c header broken edit warns", "syntax error" in res, res)


def test_python_check_names_its_interpreter_and_skips_a_bom():
    # The check runs on chad's interpreter, which may be older than the project's (a
    # 3.12 `type` statement on a 3.11 chad): the warning says which, so the model can
    # weigh it. A byte-order mark is an encoding signature every interpreter skips.
    p = _tmp("v.py", "")
    res = tool_write(p, "def f(:\n")
    check("warning names the interpreter", "under Python 3." in res, res)
    p = _tmp("b.py", "")
    res = tool_write(p, "\ufeffimport os\nprint(os.sep)\n")
    check("bom write silent", "warning" not in res, res)
    res = tool_edit(p, "print(os.sep)", "print(os.sep, 1)")
    check("bom edit silent", "warning" not in res, res)


def test_jsonc_by_convention_is_not_policed():
    for rel in ("tsconfig.json", "tsconfig.build.json", "jsconfig.json", ".eslintrc.json",
                os.path.join(".vscode", "settings.json"), "demo.code-workspace"):
        check(f"{rel} unpoliced", _code_lang(os.path.join("/w", rel)) is None, rel)
    check("package.json still policed", _code_lang("/w/package.json") == "json")
    p = _tmp("package.json", "")
    res = tool_write(p, '{"name": "x",}\n')
    check("broken package.json warns", "syntax error" in res, res)


def test_valid_files_in_many_languages_are_silent():
    """Idiomatic, valid files in every language a coding agent is likely to write, each
    landed as a fresh write (baseline zero, the strictest case). A grammar that rejects
    one of these belongs in `syntaxgate._UNRELIABLE_LANGS`, with the file kept under
    `grammar-gaps/` as the evidence; those files must stay unpoliced. Each is stored
    with a `.txt` suffix so no Python tooling mistakes the corpus for code; the real
    name is what the gate sees."""
    d = tempfile.mkdtemp(prefix="syntaxgate_corpus_")
    seen = 0
    for root, _, files in os.walk(_CORPUS):
        gap = os.path.basename(root) == "grammar-gaps"
        for stored in sorted(files):
            name = stored.removesuffix(".txt")
            with open(os.path.join(root, stored), "rb") as f:
                text = f.read().decode("utf-8", errors="replace")
            p = os.path.join(d, "gaps" if gap else "ok", name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            res = tool_write(p, text)
            check(f"{name} fresh write silent", "warning" not in res, res)
            if gap:
                check(f"{name} is unpoliced", _code_lang(p) is None, _code_lang(p))
            seen += 1
    check("corpus present", seen > 50, seen)


if __name__ == "__main__":
    test_python()
    test_cpp_header_judged_by_the_grammar_that_fits()
    test_python_check_names_its_interpreter_and_skips_a_bom()
    test_jsonc_by_convention_is_not_policed()
    test_valid_files_in_many_languages_are_silent()
    test_plain_text_never_policed()
    test_tree_sitter_delta()
    test_opt_out()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
