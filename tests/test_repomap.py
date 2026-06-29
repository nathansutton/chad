"""Tests for the tree-sitter repo map (repomap.py), focused on the file-scan
memoization added by plan 028.

`RepoMap._code_files()` walks the whole tree (`root/**/*`) plus an isfile/getsize/
language-detect on every entry, and it runs on *every* find_symbol / view_symbol /
find_refs / repo_map call. The per-file parse is already mtime-cached (`_extract`),
but the directory walk was not — so an agent doing a handful of symbol lookups
re-globbed the entire repo each time. Since `repomap.service()` is a cwd-cached
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

    # === 2. single-scan: glob the tree once across two lookups ===================
    rm2 = RepoMap(root)
    calls = {"n": 0}
    real_glob = repomap._glob.glob

    def counting_glob(*a, **k):
        calls["n"] += 1
        return real_glob(*a, **k)

    repomap._glob.glob = counting_glob
    try:
        rm2.find_symbol("alpha_func")
        rm2.find_symbol("beta_func")
    finally:
        repomap._glob.glob = real_glob

    check("two lookups glob the tree exactly once (memoized)",
          calls["n"] == 1, f"glob called {calls['n']} times")
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


if __name__ == "__main__":
    test_repomap()
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
