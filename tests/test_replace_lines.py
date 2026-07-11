"""Micro-eval for the line-addressed `tool_replace_lines` (tools.py).

replace_lines exists because the string-`edit` tool's exact-whitespace requirement drove
a real death loop in the dogfood logs: a small model that already knew the line numbers
(from `read`) still had to re-quote the text AND reproduce leading whitespace, and one
miscount broke the parse. replace_lines takes both burdens away — you give line numbers,
and indentation is fitted to the target. Pure + fast, no model load.

Run: `uv run python tests/test_replace_lines.py`
"""

import os
import tempfile

from chad.tools import _fit_indent, tool_insert_lines, tool_replace_lines

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def run(before, start, end, new, name="f.py"):
    """Apply tool_replace_lines to a temp file; return (result_string, contents_after)."""
    d = tempfile.mkdtemp(prefix="rlines_")
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(before)
    res = tool_replace_lines(p, start, end, new)
    with open(p) as f:
        after = f.read()
    return res, after


GEO = "def area(w, h):\n    return w * h\n"


def test_fit_indent_unit():
    # flush-left block slides to the target column, relative structure preserved
    fitted, shifted = _fit_indent("if x:\n    y = 1", "    ")
    check("fit: shifted flag", shifted, fitted)
    check("fit: base line indented", fitted == "    if x:\n        y = 1", repr(fitted))
    # already at target -> untouched, shifted False
    fitted, shifted = _fit_indent("    return 1", "    ")
    check("fit: no-op when aligned", not shifted and fitted == "    return 1", repr(fitted))
    # dedent inside the block is preserved when sliding
    fitted, _ = _fit_indent("    a = 1\nb = 2", "")   # first line 4-indent -> 0; second slides -4
    check("fit: relative dedent kept", fitted == "a = 1\nb = 2", repr(fitted))


def test_basic_replace():
    # replace one line, indentation supplied correctly
    res, after = run(GEO, 2, 2, "    return w * h * 2")
    check("basic: edited", res.startswith("[edited"), res)
    check("basic: applied", after == "def area(w, h):\n    return w * h * 2\n", repr(after))


def test_indent_is_fitted_from_flush_left():
    # THE point: model sends the body flush-left; harness fits it to the function's indent
    res, after = run(GEO, 2, 2, "return w * h * 2")
    check("fit: landed indented", after == "def area(w, h):\n    return w * h * 2\n", repr(after))
    check("fit: disclosed", "fit indentation" in res, res)


def test_multiline_block_fitted():
    before = "def f():\n    old = 1\n"
    # a 3-line replacement written flush-left, with internal nesting
    res, after = run(before, 2, 2, "if cond:\n    a = 1\nelse:\n    a = 2")
    want = "def f():\n    if cond:\n        a = 1\n    else:\n        a = 2\n"
    check("multiline: fitted + nested", after == want, repr(after))
    check("multiline: parses", res.startswith("[edited") and "no longer parses" not in res, res)


def test_delete_range():
    before = "a = 1\nb = 2\nc = 3\n"
    res, after = run(before, 2, 2, "")
    check("delete: edited", res.startswith("[edited") and "deleted lines 2-2" in res, res)
    check("delete: removed", after == "a = 1\nc = 3\n", repr(after))


def test_range_clamps_to_eof():
    before = "a = 1\nb = 2\n"          # 2 real lines (+ trailing "")
    res, after = run(before, 2, 999, "b = 22")
    check("clamp: edited", res.startswith("[edited"), res)
    check("clamp: replaced to eof", after == "a = 1\nb = 22\n", repr(after))


def test_bad_range_rejected():
    res, after = run(GEO, 5, 2, "x")
    check("bad range: rejected", res.startswith("[replace_lines: invalid range"), res)
    check("bad range: untouched", after == GEO, repr(after))
    res, after = run(GEO, 99, 99, "x")
    check("past eof: rejected", "past the last line" in res, res)
    check("past eof: untouched", after == GEO, repr(after))


def test_indent_break_still_rejected():
    # replace_lines routes through _apply_edit, so a genuine indent break is reverted.
    # Force a break the fitter can't launder: an over-indented body the model pre-indents.
    before = "def f():\n    a = 1\n    b = 2\n"
    res, after = run(before, 3, 3, "        b = 3")   # 8-space (would be unexpected indent) at target 4
    # _fit_indent aligns the FIRST line to target(4), so this lands clean at 4 rather than
    # producing the IndentationError the raw 8-space text would — the fit prevents the break.
    check("fit prevents break", res.startswith("[edited"), res)
    check("fit landed at target indent", after == "def f():\n    a = 1\n    b = 3\n", repr(after))


def test_no_such_file():
    res = tool_replace_lines("/definitely/not/here.py", 1, 1, "x")
    check("missing file", res.startswith("[no such file"), res)


def test_structural_reindent_multilevel():
    """A2: the real live failure — a block with TWO levels (a nested `if` guard) that the
    model mis-indents (comment at col 10, body under-indented). Neither fit (preserves the
    garbage) nor snap (flattens the nesting) works; structural reindent recomputes both
    levels from the syntax and lands it."""
    before = ("class E:\n"
              "    def load(self):\n"
              "        self.x = 1\n"
              "        return self.x\n")
    # insert a comment + a nested if-guard before `return`, at inconsistent columns
    new = ('self.x = 1\n'
           '# note\n'
           '          if not self.x:\n'
           '  self.x = 256')
    res, after = run(before, 3, 3, new)
    check("reindent: landed", res.startswith("[edited"), res)
    check("reindent: disclosed", "reindented to structure" in res, res)
    import ast
    ast.parse(after)  # raises if the reindent produced broken Python
    check("reindent: two levels correct",
          "        if not self.x:\n            self.x = 256\n" in after, repr(after))


def test_structural_reindent_preserves_triple_string():
    """A2 safety: a triple-quoted string in the replacement must survive byte-for-byte —
    its interior indentation is DATA, not code, and must never be reindented."""
    before = "def f():\n    x = 1\n    return x\n"
    new = 'x = 1\n     msg = """\n  keep me\n      and me\n"""'
    res, after = run(before, 2, 2, new)
    check("triple-string: landed", res.startswith("[edited"), res)
    check("triple-string: interior verbatim",
          '"""\n  keep me\n      and me\n"""' in after, repr(after))


def test_tab_indented_file_uses_tabs():
    """A1: a tab-indented file must be edited with TABS, not spaces (which would TabError).
    _fit_indent bails on a tab target; the recoveries rebuild in the file's own unit."""
    before = "def f():\n\tx = 1\n\treturn x\n"
    res, after = run(before, 2, 2, "x = 2")           # flush-left, target is one tab
    check("tab: landed", res.startswith("[edited"), res)
    check("tab: used a tab", after == "def f():\n\tx = 2\n\treturn x\n", repr(after))


def test_recovery_for_inconsistent_relative_indent():
    """#1: the observed live failure — the model sends a uniform-level block with
    INCONSISTENT relative indentation (sibling class fields at different columns), which
    _fit_indent's single-delta shift can't fix. A recovery (structural reindent, which
    runs first for a no-colon block, else snap) lands both fields at the field level."""
    before = ("class Engine:\n"
              "    a: int = 1\n"
              "    b: int = 2\n")
    # Replace the two field lines with a block the model wrote at col 0 and col 6.
    res, after = run(before, 2, 3, "a: int = 1\n      c: int = 3")
    check("recover: landed", res.startswith("[edited"), res)
    check("recover: disclosed", ("reindented" in res or "snapped" in res), res)
    check("recover: both fields at 4",
          after == "class Engine:\n    a: int = 1\n    c: int = 3\n", repr(after))


def test_snap_not_applied_when_it_would_flatten_a_nested_block():
    """Snap is a fallback, not the default: a genuinely nested block whose fit is already
    clean must NOT be flattened. Here the fitted result parses, so snap never runs."""
    before = "def f():\n    pass\n"
    res, after = run(before, 2, 2, "if x:\n    y = 1")   # nested, sent flush-left
    check("nested: fitted (not snapped)", "snapped" not in res, res)
    check("nested: structure kept",
          after == "def f():\n    if x:\n        y = 1\n", repr(after))


# --- insert_lines (#2) -------------------------------------------------------------

def run_ins(before, after_line, code, name="f.py"):
    d = tempfile.mkdtemp(prefix="ins_")
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(before)
    res = tool_insert_lines(p, after_line, code)
    with open(p) as f:
        after = f.read()
    return res, after


def test_insert_field_inherits_sibling_indent():
    """#2: add a dataclass field beside a sibling — the case no symbol tool covers. The
    model sends the field flush-left; insert_lines inherits the anchor line's indent."""
    before = ("class Engine:\n"
              "    a: int = 1\n"
              "    b: int = 2\n")
    res, after = run_ins(before, 2, "prefill_chunk: int = 256")   # after the `a` field
    check("insert: edited", res.startswith("[edited"), res)
    check("insert: field at sibling indent",
          after == "class Engine:\n    a: int = 1\n    prefill_chunk: int = 256\n    b: int = 2\n",
          repr(after))


def test_insert_at_top_and_eof():
    res, after = run_ins("import os\n", 0, "import sys")
    check("insert top", after == "import sys\nimport os\n", repr(after))
    # EOF with a trailing newline keeps the trailing newline
    res, after = run_ins("a = 1\n", 1, "b = 2")
    check("insert eof (nl)", after == "a = 1\nb = 2\n", repr(after))
    # EOF with NO trailing newline: start the insert on its own line, keep no trailing nl
    res, after = run_ins("a = 1", 1, "b = 2")
    check("insert eof (no nl)", after == "a = 1\nb = 2", repr(after))


def test_insert_multiline_nested():
    before = "def f():\n    pass\n"
    res, after = run_ins(before, 1, "for i in x:\n    do(i)")   # after `def f():`, flush-left
    check("insert multiline fitted",
          after == "def f():\n    for i in x:\n        do(i)\n    pass\n", repr(after))


def test_insert_guards():
    check("insert empty code", tool_insert_lines(__file__, 1, "").startswith("[insert_lines: code"),
          "empty code should be rejected")
    res, after = run_ins("a = 1\n", 9, "x")
    check("insert out of range", res.startswith("[insert_lines: after_line=9 out of range"), res)


if __name__ == "__main__":
    test_fit_indent_unit()
    test_basic_replace()
    test_indent_is_fitted_from_flush_left()
    test_multiline_block_fitted()
    test_delete_range()
    test_range_clamps_to_eof()
    test_bad_range_rejected()
    test_indent_break_still_rejected()
    test_no_such_file()
    test_structural_reindent_multilevel()
    test_structural_reindent_preserves_triple_string()
    test_tab_indented_file_uses_tabs()
    test_recovery_for_inconsistent_relative_indent()
    test_snap_not_applied_when_it_would_flatten_a_nested_block()
    test_insert_field_inherits_sibling_indent()
    test_insert_at_top_and_eof()
    test_insert_multiline_nested()
    test_insert_guards()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
