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


def test_repomap(tmp_path=None):
    # Allow standalone (__main__) invocation without pytest's tmp_path fixture.
    if tmp_path is None:
        import pathlib
        import tempfile
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_test_"))

    root = _make_repo(tmp_path)

    # === 1. correctness: lookups still resolve the right symbol ==================
    rm = RepoMap(root)
    fs = rm.find_symbol("alpha_func")
    check("find_symbol locates alpha_func", "alpha_func" in fs and "alpha.py" in fs, fs)

    vs = rm.view_symbol("beta_func")
    check("view_symbol returns beta_func body",
          "beta_func" in vs and "alpha_func(2)" in vs, vs)

    vs_cls = rm.view_symbol("AlphaClass")
    check("view_symbol returns AlphaClass body",
          "class AlphaClass" in vs_cls, vs_cls)

    # === 2. single-scan: walk the tree once across two lookups ===================
    rm2 = RepoMap(root)
    calls = {"n": 0}
    real_walk = repomap.os.walk

    def counting_walk(*a, **k):
        calls["n"] += 1
        return real_walk(*a, **k)

    repomap.os.walk = counting_walk
    try:
        rm2.find_symbol("alpha_func")
        rm2.find_symbol("beta_func")
    finally:
        repomap.os.walk = real_walk

    check("two lookups walk the tree exactly once (memoized)",
          calls["n"] == 1, f"os.walk called {calls['n']} times")
    check("_files cache is populated after a completed scan",
          rm2._files is not None and len(rm2._files) == 2,
          repr(rm2._files))

    # === 3. no-cache-on-interrupt: a stopped scan is not cached ==================
    rm3 = RepoMap(root)
    interrupted = rm3._code_files(should_stop=lambda: True)
    check("interrupted scan returns a (partial) list", isinstance(interrupted, list))
    check("interrupted scan is NOT cached (self._files stays None)",
          rm3._files is None)

    # a subsequent normal call still computes and caches the full list
    full = rm3._code_files()
    check("normal call after interrupt computes the full list",
          len(full) == 2, repr(full))
    check("normal call after interrupt caches the result",
          rm3._files is not None and len(rm3._files) == 2)

    # and once cached, the cached object is reused identically (not recomputed)
    check("second normal call returns the same cached object",
          rm3._code_files() is rm3._files)


def test_repo_map_edge_aggregation():
    """Pin the rank-graph memory fix: raw references collapse to ONE weighted edge
    per (referencer, definer) file pair, and an identifier defined in more than
    _MAX_DEFINERS files is excluded from the graph entirely. The naive per-reference
    edge list was definers × references objects — 32M edges / 10 GB on an 11k-file
    repo — which starved Metal of memory and aborted the whole process mid-session."""
    from collections import Counter

    defines = {"shared": {"lib.py"},
               "everywhere": {f"gen{i:02d}.py"
                              for i in range(repomap._MAX_DEFINERS + 1)}}
    references = {"shared": Counter({"caller.py": 9}),
                  "everywhere": Counter({"refs_everywhere.py": 3})}

    edges = repomap._build_edges(defines, references)
    check("9 raw references collapse to one aggregated edge",
          list(edges) == [("caller.py", "lib.py")], repr(edges))
    check("aggregated edge weight is sqrt of the per-file reference count",
          abs(edges[("caller.py", "lib.py")] - 3.0) < 1e-9, repr(edges))
    check("an ident defined in > _MAX_DEFINERS files contributes no edges",
          not any(v.startswith("gen") for _u, v in edges), repr(edges))
    check("self-references (definer file citing itself) add no edge",
          not repomap._build_edges({"f": {"a.py"}}, {"f": Counter({"a.py": 4})}))


def test_rank_files():
    """The rustworkx ranking: referenced files outrank unreferenced ones, and a
    personalization seed pulls rank toward its own reference targets."""
    rels = ["hub.py", "leaf.py", "island.py"]
    edges = {("leaf.py", "hub.py"): 2.0}
    ranks = repomap._rank_files(rels, edges, {})
    check("a referenced file outranks an unreferenced one",
          ranks["hub.py"] > ranks["island.py"], repr(ranks))

    seeded = repomap._rank_files(rels, edges, {"leaf.py": 1.0})
    check("rank flows from the seed to its reference target",
          seeded["hub.py"] > seeded["island.py"], f"{ranks} -> {seeded}")
    check("a file unrelated to the seed loses rank under personalization",
          seeded["island.py"] < ranks["island.py"], f"{ranks} -> {seeded}")

    check("empty graph ranks nothing", repomap._rank_files([], {}, {}) == {})


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
        out1 = RepoMap(str(repo)).repo_map()
        check("first scan sees the fixture symbols", "func_00" in out1, out1[:200])
        cache_files = list(pathlib.Path(cache_dir).glob("*.pkl"))
        check("scan above the save threshold persists a cache file",
              len(cache_files) == 1, repr(cache_files))

        # A fresh instance must be served from disk: make parsing impossible.
        real_qc = repomap.QueryCursor

        def boom(*a, **k):
            raise AssertionError("parse attempted despite warm disk cache")

        repomap.QueryCursor = boom
        try:
            out2 = RepoMap(str(repo)).repo_map()
        finally:
            repomap.QueryCursor = real_qc
        check("warm scan is served entirely from the disk cache",
              "func_00" in out2 and "func_33" in out2, out2[:200])

        # mtime invalidation: rewrite one file, its new symbol must appear.
        target = repo / "mod00.py"
        target.write_text("def renamed_func():\n    pass\n")
        import os as _os
        _os.utime(target, (1e9, 1e9))  # force an mtime change even on coarse clocks
        out3 = RepoMap(str(repo)).repo_map()
        check("a changed file is re-parsed (new symbol appears)",
              "renamed_func" in out3, out3[:300])
        check("the changed file's old symbol is gone", "func_00" not in out3)
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


def test_aggregate_memoization(tmp_path=None):
    """The incremental rank-graph tables: unchanged files aren't re-folded between
    repo_map calls, an edited file's new symbols replace its old ones, and a deleted
    file's symbols drop out of the map."""
    import os as _os
    import pathlib
    import tempfile
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_agg_"))
    (tmp_path / "stable.py").write_text("def stays():\n    pass\n")
    (tmp_path / "volatile.py").write_text("def before_edit():\n    pass\n")
    (tmp_path / "doomed.py").write_text("def dead_symbol():\n    pass\n")

    rm = RepoMap(str(tmp_path))
    out1 = rm.repo_map()
    check("initial map has all three files' symbols",
          all(s in out1 for s in ("stays", "before_edit", "dead_symbol")), out1)
    contrib = rm._agg[0]
    stable_entry = contrib[str(tmp_path / "stable.py")]

    target = tmp_path / "volatile.py"
    target.write_text("def after_edit():\n    pass\n")
    _os.utime(target, (1e9, 1e9))  # force an mtime change even on coarse clocks
    (tmp_path / "doomed.py").unlink()

    out2 = rm.repo_map()
    check("edited file's new symbol appears", "after_edit" in out2, out2)
    check("edited file's old symbol is unfolded", "before_edit" not in out2, out2)
    check("deleted file's symbols drop out of the map",
          "dead_symbol" not in out2, out2)
    check("an unchanged file's contribution is reused, not re-folded",
          contrib[str(tmp_path / "stable.py")] is stable_entry)


def test_disambig_budget_and_cache(tmp_path=None):
    """Disambiguation annotations run under a shared wall-clock budget (a slow
    language server must not turn a candidate list into a minutes-long stall) and
    are cached per definition site (a repeated view_symbol used to re-pay every
    LSP lookup — measured 79s on a big mixed repo)."""
    import pathlib
    import tempfile
    if tmp_path is None:
        tmp_path = pathlib.Path(tempfile.mkdtemp(prefix="repomap_disambig_"))
    for i in range(6):
        (tmp_path / f"m{i}.py").write_text("def ambiguous():\n    pass\n")

    rm = RepoMap(str(tmp_path))
    calls = {"n": 0}

    def fake_summary(h, timeout=None):
        calls["n"] += 1
        return "  — used by fake.py"

    real_summary, real_budget = rm._ref_summary, repomap._DISAMBIG_BUDGET_S
    rm._ref_summary = fake_summary
    try:
        out = rm.view_symbol("ambiguous")
        check("ambiguous lookup lists every candidate",
              out.count("ambiguous") >= 6, out)
        check("annotations come from the summary hook",
              "used by fake.py" in out, out)
        check("one summary call per candidate", calls["n"] == 6, calls["n"])

        repomap._DISAMBIG_BUDGET_S = -1.0  # budget already exhausted
        calls["n"] = 0
        out2 = rm.view_symbol("ambiguous")
        check("an exhausted budget still lists all candidates, un-annotated",
              out2.count("m0.py") == 1 and "used by" not in out2, out2)
        check("no summary calls once the budget is spent", calls["n"] == 0)
        check("unreached candidates are cached blank (no re-pay on repeat)",
              len(rm._refsum_cache) >= 6, len(rm._refsum_cache))
    finally:
        rm._ref_summary, repomap._DISAMBIG_BUDGET_S = real_summary, real_budget

    # the real _ref_summary caches per (site, mtime): a second identical call must
    # not hit the LSP service again
    import os as _os

    import chad.lsp as lsp_mod

    class _CountingSvc:
        def __init__(self):
            self.n = 0
            self.root = _os.path.abspath(_os.getcwd())  # so service() won't rebind

        def references_decorative(self, rel, row, col, timeout=None):
            self.n += 1
            return [(f"user{self.n}.py", 1, 1)]

    svc = _CountingSvc()
    real_svc = lsp_mod._SERVICE
    lsp_mod._SERVICE = svc
    try:
        rm2 = RepoMap(str(tmp_path))
        h = rm2._find_defs("ambiguous")[0]
        n1 = rm2._ref_summary(h)
        n2 = rm2._ref_summary(h)
        check("ref summary is cached per site (one LSP call for two lookups)",
              svc.n == 1 and n1 == n2 == "  — used by user1.py", (svc.n, n1, n2))
    finally:
        lsp_mod._SERVICE = real_svc


if __name__ == "__main__":
    test_repomap()
    test_repo_map_edge_aggregation()
    test_rank_files()
    test_disk_cache()
    test_parallel_extract()
    test_aggregate_memoization()
    test_disambig_budget_and_cache()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


# --- wheel-less platforms ------------------------------------------
# tree-sitter-language-pack is a native wheel. On a TB2 container (emulated amd64) uv
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
