"""Polyglot coverage for the tree-sitter substrate (tests/fixtures/polyglot).

The repo map claims language-agnostic symbol intelligence via the language pack's
grammars, yet until this file every repo-map fixture was Python. Each fixture
language carries the same three deliberate cases:

  * a class method whose bare name collides with a free function elsewhere
    (`Engine.process` vs `process`) — the qualified-path case,
  * a cross-file reference through an import (`helper` defined in lib, called in
    app) — the precision case,
  * the same name defined in two files — the disambiguation case.

First measurement (2026-07-29) found the pack's tags queries broken or missing for
4 of 12 languages — typescript/tsx (no query at all), php (query names a node type
the current grammar renamed), c (no reference captures; header prototypes tagged
as definitions), cpp (out-of-class `Engine::process` invisible) — fixed by chad's
`repomap._TAGS_OVERRIDE`, which also adds bash outright (the pack ships no bash
query; 89 of 91 terminal-bench-2.1 tasks contain shell). This test pins all 13
languages so a language-pack bump that breaks a grammar's tags surfaces here, not
in an agent run.

No model, no language server, no network; runs in the fast gate.
"""

import os

import pytest

from chad.repomap import RepoMap

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "polyglot")

# lang -> (helper fn, colliding name, caller fn, container symbol, lib file, app file)
LANGS = {
    "python": ("helper", "process", "run", "Engine", "lib.py", "app.py"),
    "typescript": ("helper", "process", "run", "Engine", "lib.ts", "app.ts"),
    "javascript": ("helper", "process", "run", "Engine", "lib.js", "app.js"),
    "go": ("Helper", "Process", "Run", "Engine", "lib.go", "app.go"),
    "rust": ("helper", "process", "run", "Engine", "lib.rs", "app.rs"),
    "java": ("helper", "process", "run", "Engine", "Engine.java", "App.java"),
    "ruby": ("helper", "process", "run", "Engine", "lib.rb", "app.rb"),
    "csharp": ("Helper", "Process", "Run", "Engine", "Lib.cs", "App.cs"),
    "php": ("helper", "process", "run", "Engine", "lib.php", "app.php"),
    "kotlin": ("helper", "process", "run", "Engine", "Lib.kt", "App.kt"),
    "c": ("helper", "process", "run", "engine_process", "lib.c", "app.c"),
    "cpp": ("helper", "process", "run", "Engine", "lib.cpp", "app.cpp"),
    "bash": ("helper", "process", "run", "engine_process", "lib.sh", "app.sh"),
}


def _map_for(lang):
    rm = RepoMap(os.path.join(FIXTURES, lang))
    rm._disk_checked = True  # hermetic: never read/write the user's real tags cache
    return rm


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_definitions_extracted(lang):
    """Every fixture language yields its definitions: the container symbol, the
    method/collider in lib, and the free function + caller in app."""
    helper, process, run, container, lib, app = LANGS[lang]
    rm = _map_for(lang)
    by_file = {}
    for f in rm._code_files():
        defs, _ = rm._extract(f)
        by_file[rm._rel(f)] = {d.name for d in defs}
    everywhere = set().union(*by_file.values())
    assert container in everywhere, by_file  # cpp: class in lib.hpp, methods in lib.cpp
    assert process in by_file[lib], by_file
    assert process in by_file[app], by_file
    assert run in by_file[app], by_file


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_find_and_view_symbol(lang):
    helper, _process, run, _container, lib, app = LANGS[lang]
    rm = _map_for(lang)
    fs = rm.find_symbol(helper)
    assert lib in fs, fs
    vs = rm.view_symbol(run)
    assert vs.startswith(f"{app}:"), vs
    assert process_call_present(vs, lang), vs


def process_call_present(view, lang):
    """The viewed `run` body must contain the call into Engine's method — the span
    is real code, not a header-only stub."""
    needle = {"go": "e.Process(", "csharp": ".Process(", "c": "engine_process(",
              "php": "->process(", "bash": "engine_process "}.get(lang, ".process(")
    return needle in view


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_cross_file_refs(lang):
    """`helper` is defined in lib and called in app: find_refs must return the
    cross-file hit (name-match honesty label is fine; missing the hit is not)."""
    helper, _process, _run, _container, _lib, app = LANGS[lang]
    rm = _map_for(lang)
    fr = rm.find_refs(helper)
    assert app in fr, fr


@pytest.mark.parametrize("lang", sorted(LANGS))
def test_same_name_two_files_disambiguates(lang):
    """The colliding name is defined in two files; a bare view_symbol must return
    the disambiguation listing naming both, not silently pick one."""
    _helper, process, _run, _container, lib, app = LANGS[lang]
    rm = _map_for(lang)
    out = rm.view_symbol(process)
    assert "pass path= to disambiguate" in out, out
    assert lib in out and app in out, out
