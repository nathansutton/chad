"""Characterization tests for symbols.py — the jedi-backed symbol EDITOR that writes
to source files. Previously ZERO direct coverage. test_edit.py is the text-edit
analogue; this is the symbol-edit one.

A bug in `_span`/`_apply` silently edits the wrong line range (clobbering an adjacent
symbol); a bug in `_locate_one`/`_matches` targets a free function when a method was
meant. These pin: replace lands on exactly the named span, qualified `Class/method`
targeting works, insert keeps the file parseable, and the not-found / empty-content
guards leave the file untouched. jedi is a hard dependency so these always run — no
model gate.

Run: `uv run python test_symbols.py`
"""

import os
import tempfile

from chad.symbols import SymbolService

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


SRC = """def alpha():
    return 1

class C:
    def beta(self):
        return 2

def gamma():
    return 3
"""


def _fixture(tmp_path):
    """Write the fixture module and return (SymbolService, file_path)."""
    fp = os.path.join(str(tmp_path), "m.py")
    with open(fp, "w") as f:
        f.write(SRC)
    return SymbolService(str(tmp_path)), fp


def test_replace_lands_on_one_symbol(tmp_path):
    svc, fp = _fixture(tmp_path)
    res = svc.replace_symbol("alpha", "def alpha():\n    return 99", path=fp)
    check("replace returns a diff/label", res.startswith("[replaced"), res)
    after = open(fp).read()
    # alpha rewritten...
    check("alpha rewritten to 99", "return 99" in after, after)
    # ...and ONLY alpha — beta and gamma are byte-for-byte intact.
    check("beta untouched", "    def beta(self):\n        return 2" in after, after)
    check("gamma untouched", "def gamma():\n    return 3" in after, after)
    # the original alpha body is gone (not duplicated/appended)
    check("old alpha body gone", "def alpha():\n    return 1" not in after, after)
    check("still parses", _parses(after, fp))


def test_qualified_name_targets_method(tmp_path):
    svc, fp = _fixture(tmp_path)
    # 'C/beta' must hit the method, not search free functions; rewrite its body.
    res = svc.replace_symbol("C/beta", "    def beta(self):\n        return 222", path=fp)
    check("qualified replace returns label", res.startswith("[replaced"), res)
    check("qualified label names C.beta", "C.beta" in res, res)
    after = open(fp).read()
    check("beta rewritten to 222", "return 222" in after, after)
    # the surrounding free functions are untouched
    check("alpha untouched by C/beta", "def alpha():\n    return 1" in after, after)
    check("gamma untouched by C/beta", "def gamma():\n    return 3" in after, after)
    check("still parses after method replace", _parses(after, fp))


def test_insert_after_places_and_parses(tmp_path):
    svc, fp = _fixture(tmp_path)
    res = svc.insert_symbol("alpha", "def inserted():\n    return 0",
                            where="after", path=fp)
    check("insert returns label", res.startswith("[inserted after"), res)
    after = open(fp).read()
    check("inserted function present", "def inserted():" in after, after)
    # it sits after alpha and before the class C (i.e. between them)
    check("placed after alpha", after.index("def alpha") < after.index("def inserted"), after)
    check("placed before class C", after.index("def inserted") < after.index("class C"), after)
    # alpha itself survived intact
    check("alpha still intact after insert", "def alpha():\n    return 1" in after, after)
    check("file still parses after insert", _parses(after, fp))


def test_not_found_leaves_file_untouched(tmp_path):
    svc, fp = _fixture(tmp_path)
    res = svc.replace_symbol("nope", "def nope(): return 0", path=fp)
    check("not-found message", "symbol not found" in res, res)
    check("file unchanged on not-found", open(fp).read() == SRC)


def test_empty_content_refused(tmp_path):
    svc, fp = _fixture(tmp_path)
    res = svc.replace_symbol("alpha", "   ", path=fp)
    check("empty-content refused", "refusing to replace a symbol with empty content" in res, res)
    check("file unchanged on empty content", open(fp).read() == SRC)


def _parses(src, fp):
    try:
        compile(src, fp, "exec")
        return True
    except SyntaxError:
        return False


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        test_replace_lands_on_one_symbol(d)
    with tempfile.TemporaryDirectory() as d:
        test_qualified_name_targets_method(d)
    with tempfile.TemporaryDirectory() as d:
        test_insert_after_places_and_parses(d)
    with tempfile.TemporaryDirectory() as d:
        test_not_found_leaves_file_untouched(d)
    with tempfile.TemporaryDirectory() as d:
        test_empty_content_refused(d)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
