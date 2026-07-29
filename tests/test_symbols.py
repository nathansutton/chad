"""Characterization tests for symbols.py — the symbol EDITOR that writes to source
files. test_edit.py is the text-edit analogue; this is the symbol-edit one.

A bug in `_locate_one`/`_apply` silently edits the wrong line range (clobbering an
adjacent symbol) or targets a free function when a method was meant. These pin:
replace lands on exactly the named span, qualified `Class/method` targeting works
(in TWO languages — the backend is one tree-sitter path for all of them, so an
asymmetry here is a regression), insert keeps the file parseable, and the
not-found / empty-content guards leave the file untouched. No model gate.

Run: `uv run python tests/test_symbols.py`
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


def test_qualified_method_vs_free_function(tmp_path):
    """A method and a free function share a bare name: the qualified path must hit
    the method, the bare name must disambiguate — in both fixture languages."""
    svc = _tree(tmp_path, {
        "shape.py": ("class Shape:\n"
                     "    def area(self):\n"
                     "        return 1\n"
                     "\n\n"
                     "def area():\n"
                     "    return 2\n"),
        "shape.js": ("class Shape {\n"
                     "  area() {\n"
                     "    return 1;\n"
                     "  }\n"
                     "}\n"
                     "\n"
                     "function area() {\n"
                     "  return 2;\n"
                     "}\n"),
    })
    for fn, comment in (("shape.py", "python"), ("shape.js", "javascript")):
        fp = os.path.join(str(tmp_path), fn)
        res = svc.replace_symbol(
            "Shape/area",
            "  def area(self):\n        return 11" if fn.endswith(".py")
            else "  area() {\n    return 11;\n  }",
            path=fp)
        check(f"{comment}: qualified replace lands", res.startswith("[replaced"), res)
        check(f"{comment}: label is Shape.area", "Shape.area" in res, res)
        after = open(fp).read()
        check(f"{comment}: free area() untouched", "return 2" in after, after)
        check(f"{comment}: method rewritten", "11" in after, after)
        # the bare name stays ambiguous: no silent pick between method and function
        bare = svc.replace_symbol("area", "x", path=None)
        check(f"{comment}: bare name disambiguates", "disambiguate" in bare, bare)


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


def test_qualified_never_falls_back_to_bare(tmp_path):
    """A qualified path whose container doesn't exist must be a miss, not a silent
    resolve to the bare last segment (editing the wrong symbol is corruption)."""
    svc, fp = _fixture(tmp_path)
    res = svc.replace_symbol("NoSuchClass/alpha", "def alpha():\n    return 0", path=fp)
    check("wrong qualifier is a miss", "symbol not found" in res, res)
    check("file unchanged on qualified miss", open(fp).read() == SRC)


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


def _tree(tmp_path, spec):
    """Write {relpath: content} under tmp_path and return a SymbolService on it."""
    for rel, content in spec.items():
        p = os.path.join(str(tmp_path), rel)
        with open(p, "w") as f:
            f.write(content)
    return SymbolService(str(tmp_path))


def test_mass_definer_disambiguates_capped(tmp_path):
    """A name defined in dozens of files yields a capped disambiguation listing
    (unbounded listings measured 205 lines of 'main' straight into prefill), and
    path= still resolves precisely."""
    from chad import symbols as symmod
    n = symmod._DISAMBIG_MAX_LINES + 19
    svc = _tree(tmp_path, {f"m{i:03d}.py": "def forward(x):\n    return x\n"
                           for i in range(n)})
    res = svc.replace_symbol("forward", "def forward(x):\n    return x + 1")
    check("mass definer disambiguates", f"{n} symbols named 'forward'" in res, res)
    listed = [ln for ln in res.splitlines() if ln.startswith("  m")]
    check("listing capped", len(listed) == symmod._DISAMBIG_MAX_LINES, res)
    check("overflow announced", "more)" in res.splitlines()[-1], res)
    res = svc.replace_symbol("forward", "def forward(x):\n    return x + 1",
                             path=os.path.join(str(tmp_path), "m000.py"))
    check("path= still resolves precisely", res.startswith("[replaced"), res)


def test_new_file_visible_despite_walk_memo(tmp_path):
    svc = _tree(tmp_path, {"a.py": "def alpha():\n    pass\n"})
    check("warm-up locate works", svc._locate_one("alpha", None)[0] is not None)
    with open(os.path.join(str(tmp_path), "fresh.py"), "w") as f:
        f.write("def beta():\n    pass\n")
    svc._repomap()._files_at -= 2.0  # age the memo past the re-walk guard
    hit, err = svc._locate_one("beta", None)
    check("file created after the cached walk is found", hit is not None, err)


# ---------------------------------------------------------------------------
# Multi-language editing: every language resolves through the tree-sitter repo
# map's definition spans (the same ones view_symbol shows). Skips cleanly if the
# grammar can't load.
# ---------------------------------------------------------------------------

JS = """function greet(name) {
  return "hi " + name;
}

function farewell(name) {
  return "bye " + name;
}
"""


def _js_available(svc):
    hit, _err = svc._locate_one("greet", None)
    return hit is not None


def test_replace_symbol_javascript(tmp_path):
    svc = _tree(tmp_path, {"app.js": JS})
    if not _js_available(svc):
        return  # no javascript grammar in this env; the routing is covered below
    res = svc.replace_symbol("greet", 'function greet(name) {\n  return "yo " + name;\n}')
    check("js replace lands", res.startswith("[replaced"), res)
    after = open(os.path.join(str(tmp_path), "app.js")).read()
    check("js body rewritten", '"yo " + name' in after, after)
    check("js sibling untouched", '"bye " + name' in after, after)
    res2 = svc.insert_symbol("farewell", "function extra() {\n  return 1;\n}",
                             where="after")
    check("js insert lands", res2.startswith("[inserted after"), res2)
    after2 = open(os.path.join(str(tmp_path), "app.js")).read()
    check("js insert placed after farewell",
          after2.index("function farewell") < after2.index("function extra"), after2)


def test_cross_language_collision_disambiguates(tmp_path):
    """A name defined in both a .py and a .js file is ambiguous and says so —
    the old backend silently preferred Python, which is exactly the per-language
    asymmetry the unified backend removed."""
    svc = _tree(tmp_path, {"m.py": "def greet():\n    return 1\n", "app.js": JS})
    hit, err = svc._locate_one("greet", None)
    check("collision returns no silent hit", hit is None, hit)
    check("collision disambiguates across languages",
          err and "m.py" in err and "app.js" in err, err)


def test_non_python_not_found_is_clean(tmp_path):
    svc = _tree(tmp_path, {"app.js": JS})
    res = svc.replace_symbol("no_such_symbol", "function x() {}")
    check("clean not-found for non-py", "symbol not found" in res, res)


def test_reedit_uses_fresh_parse(tmp_path):
    svc = _tree(tmp_path, {"mod.py": "def f():\n    return 1\n"})
    res = svc.replace_symbol("f", "def f():\n    return 2\ndef g():\n    return 3")
    check("first replace lands", res.startswith("[replaced"), res)
    # g() only exists post-edit; finding it proves the mtime cache evicted
    hit, err = svc._locate_one("g", None)
    check("post-edit parse is fresh", hit is not None, err)


if __name__ == "__main__":
    for fn in (test_replace_lands_on_one_symbol, test_qualified_name_targets_method,
               test_qualified_method_vs_free_function,
               test_insert_after_places_and_parses, test_not_found_leaves_file_untouched,
               test_qualified_never_falls_back_to_bare, test_empty_content_refused,
               test_mass_definer_disambiguates_capped,
               test_new_file_visible_despite_walk_memo,
               test_replace_symbol_javascript, test_cross_language_collision_disambiguates,
               test_non_python_not_found_is_clean, test_reedit_uses_fresh_parse):
        with tempfile.TemporaryDirectory() as d:
            fn(d)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
