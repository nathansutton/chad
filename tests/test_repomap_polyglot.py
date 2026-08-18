"""Polyglot coverage for the tree-sitter tags substrate (tests/fixtures/polyglot).

The bash-route ambient levers (bash_read_skeleton's one-line symbol map and its
definition pointer) claim language-agnostic extraction via the language pack's
grammars. First measurement (2026-07-29) found the pack's tags queries broken or
missing for 4 of 12 languages — typescript/tsx (no query at all), php (query
names a node type the current grammar renamed), c (no reference captures), cpp
(out-of-class `Engine::process` invisible) — fixed by chad's
`repomap._TAGS_OVERRIDE`, which also adds bash outright (the pack ships no bash
query). This pins all 13 languages so a language-pack bump that breaks a
grammar's tags surfaces here, not in an agent run.

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
def test_find_defs_resolves_the_helper(lang):
    """_find_defs — the engine behind the ambient definition pointer — locates the
    cross-file helper in every language."""
    helper, _process, _run, _container, lib, _app = LANGS[lang]
    rm = _map_for(lang)
    hits = rm._find_defs(helper)
    assert any(d.rel == lib for d in hits), hits
