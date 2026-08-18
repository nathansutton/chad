"""Tests for the tree-sitter repo map (repomap.py), focused on the file-scan
memoization in the repo-map ranker.

`RepoMap._code_files()` walks the whole tree (pruned `os.walk`) plus an isfile/
getsize/language-detect on every entry, and it runs on *every* find_symbol /
view_symbol / find_refs / repo_map call. The per-file parse is already mtime-cached
(`_extract`), but the directory walk was not — so an agent doing a handful of symbol
lookups re-walked the entire repo each time. Since `repomap.service()` is a cwd-cached
singleton that lives for the whole session, memoizing the file list on the instance
lets every lookup after the first reuse it.

This file pins three properties:

  1. Correctness — the cache doesn't break lookups (view_symbol/find_symbol still
     return the expected symbol).
  2. Single-scan — two symbol lookups glob the tree exactly once, not once per call.
  3. No-cache-on-interrupt — a scan stopped mid-walk by `should_stop` is NOT cached,
     so a later normal call still computes the full list.

No model is loaded; this runs in the fast gate.
"""

from chad import repomap
from chad.repomap import RepoMap

passed = 0
failed = 0


def check(desc, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS: {desc}")
    else:
        failed += 1
        raise AssertionError(f"{desc}{(' — ' + detail) if detail else ''}")


def _make_repo(tmp_path):
    """A tiny fixture repo with two .py files and known symbols."""
    (tmp_path / "alpha.py").write_text(
        "def alpha_func(x):\n"
        "    return x + 1\n"
        "\n"
        "class AlphaClass:\n"
        "    def method_one(self):\n"
        "        return alpha_func(1)\n"
    )
    (tmp_path / "beta.py").write_text(
        "from alpha import alpha_func\n"
        "\n"
        "def beta_func():\n"
        "    return alpha_func(2)\n"
    )
    return str(tmp_path)


def test_disk_cache(tmp_path=None):
    """The persistent tags cache: a second RepoMap instance (a new session) serves
    the scan from disk without re-parsing, and an mtime change re-parses that file."""
    import pathlib
    import tempfile
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_cache_"))
    cache_dir = str(tmp_path / "cachedir")
    repo = tmp_path / "repo"
    repo.mkdir()
    # enough files to clear the save threshold
    for i in range(repomap._CACHE_SAVE_MIN + 2):
        (repo / f"mod{i:02d}.py").write_text(f"def func_{i:02d}():\n    pass\n")

    real_dir = repomap._CACHE_DIR
    repomap._CACHE_DIR = cache_dir
    try:
        rm1 = RepoMap(str(repo))
        rm1._extract_all(rm1._code_files())
        check("first scan sees the fixture symbols",
              bool(rm1._find_defs("func_00")))
        cache_files = list(pathlib.Path(cache_dir).glob("*.pkl"))
        check("scan above the save threshold persists a cache file",
              len(cache_files) == 1, repr(cache_files))

        # A fresh instance must be served from disk: make parsing impossible.
        real_qc = repomap.QueryCursor

        def boom(*a, **k):
            raise AssertionError("parse attempted despite warm disk cache")

        repomap.QueryCursor = boom
        try:
            rm2 = RepoMap(str(repo))
            rm2._extract_all(rm2._code_files())
        finally:
            repomap.QueryCursor = real_qc
        check("warm scan is served entirely from the disk cache",
              bool(rm2._find_defs("func_00")) and bool(rm2._find_defs("func_33")))

        # mtime invalidation: rewrite one file, its new symbol must appear.
        target = repo / "mod00.py"
        target.write_text("def renamed_func():\n    pass\n")
        import os as _os
        _os.utime(target, (1e9, 1e9))  # force an mtime change even on coarse clocks
        rm3 = RepoMap(str(repo))
        rm3._extract_all(rm3._code_files())
        check("a changed file is re-parsed (new symbol appears)",
              bool(rm3._find_defs("renamed_func")))
        check("the changed file's old symbol is gone",
              not rm3._find_defs("func_00"))
    finally:
        repomap._CACHE_DIR = real_dir


def test_parallel_extract(tmp_path=None):
    """Subprocess-sharded extraction returns the same tags as the serial path, and
    an immediate should_stop kills the workers without hanging."""
    import pathlib
    import tempfile
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_par_"))
    for i in range(24):
        (tmp_path / f"m{i:02d}.py").write_text(
            f"def par_func_{i:02d}(x):\n    return x\n")

    serial = RepoMap(str(tmp_path))
    serial._disk_checked = True  # isolate from any real on-disk cache
    for f in serial._code_files():
        serial._extract(f)

    real_min = repomap._PARALLEL_MIN_FILES
    repomap._PARALLEL_MIN_FILES = 1  # force the worker path on the tiny fixture
    try:
        par = RepoMap(str(tmp_path))
        par._disk_checked = True
        par._extract_all(par._code_files())
        same = all(par._cache.get(f, (None, [], []))[1:]
                   == serial._cache.get(f, (None, [], []))[1:]
                   for f in serial._code_files())
        check("parallel extraction matches serial tags exactly", same)

        stopped = RepoMap(str(tmp_path))
        stopped._disk_checked = True
        stopped._extract_all(stopped._code_files(), should_stop=lambda: True)
        check("should_stop during parallel extraction returns without hanging", True)
    finally:
        repomap._PARALLEL_MIN_FILES = real_min


if __name__ == "__main__":
    test_repomap()
    test_repo_map_edge_aggregation()
    test_rank_files()
    test_disk_cache()
    test_parallel_extract()
    test_aggregate_memoization()
    test_disambig_budget_and_cache()
    test_definition_precise_and_fallback()
    test_definition_is_wired_into_the_toolset()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


# --- wheel-less platforms ------------------------------------------
# tree-sitter-language-pack is a native wheel. On a benchmark container (emulated amd64) uv
# fell back to a Rust source build, it failed, and the module-level import took ALL of
# chad down -- the qemu-startup trial errored before the agent ran one step.

def test_repomap_degrades_when_tree_sitter_is_absent(monkeypatch, tmp_path):
    """With tlp unavailable, symbol ranking yields nothing but nothing raises."""
    from chad import repomap
    monkeypatch.setattr(repomap, "tlp", None)
    monkeypatch.setattr(repomap, "Parser", None)
    monkeypatch.setattr(repomap, "Query", None)
    monkeypatch.setattr(repomap, "QueryCursor", None)

    rm = repomap.RepoMap(str(tmp_path))
    assert rm.lang_for(str(tmp_path / "a.py")) is None
    assert rm._lang_tools("python") is None


def test_repomap_and_tools_import_without_the_tree_sitter_wheel(monkeypatch):
    """The regression that matters: `tools` (bash/read/edit) imports `repomap` at module
    scope, so a missing tree-sitter wheel used to make ALL of chad unimportable."""
    import builtins
    import importlib

    real_import = builtins.__import__

    def _no_tree_sitter(name, *a, **kw):
        if name.startswith("tree_sitter"):
            raise ImportError(f"no wheel for {name} on this platform")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_tree_sitter)
    import chad.repomap
    import chad.tools
    try:
        reloaded = importlib.reload(chad.repomap)
        assert reloaded.tlp is None, "import guard did not engage"
        assert reloaded.Parser is None
        importlib.reload(chad.tools)      # the chain that actually broke the trial
    finally:
        monkeypatch.setattr(builtins, "__import__", real_import)
        importlib.reload(chad.repomap)    # restore for the rest of the session
        importlib.reload(chad.tools)
