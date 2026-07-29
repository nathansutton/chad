"""Context-economy metric for the symbolic stack (plan: tokens-to-target).

The symbolic stack is the context engine: its quality is how many tokens of tool
output the model must ingest to reach a correct edit. For each polyglot fixture
language this measures three fixed navigation tasks:

  qualified  resolve the Engine method whose bare name collides with a free
             function ("Engine/process"). One tool result if qualified paths
             resolve; a disambiguation listing + a second call if they don't.
  refs       find_refs on the cross-file helper: precise (language server) or
             name-match, and how many tokens the listing costs.
  navigate   find_symbol(helper) then view_symbol(run): the basic read loop.

Tokens use repomap._estimate_tokens, the same accounting the map budgets with, so
numbers are comparable to repo_map's own budget. No model load, no network.

Run: uv run python benchmarks/ctxengine/measure.py [--json]
"""

import json
import os
import sys
import time

from chad.repomap import RepoMap, _estimate_tokens

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, "tests", "fixtures", "polyglot")

# lang -> (helper, collider, caller, container, lib file, app file)
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
}


def _task_qualified(rm, names):
    """Reach the Engine method's body via its qualified name. If the first call
    comes back as a disambiguation listing, a real model must read it and issue a
    second, path-qualified call — both count."""
    helper, collider, run, container, lib, app = names
    calls, tokens = 0, 0
    out = rm.view_symbol(f"{container}/{collider}")
    calls += 1
    tokens += _estimate_tokens(out)
    resolved_directly = "disambiguate" not in out and "not found" not in out
    if not resolved_directly:
        out = rm.view_symbol(collider, path=os.path.join(rm.root, lib))
        calls += 1
        tokens += _estimate_tokens(out)
    ok = f"{lib}:" in out
    return {"tokens": tokens, "calls": calls, "qualified_resolves": resolved_directly,
            "ok": ok}


def _task_refs(rm, names):
    helper = names[0]
    out = rm.find_refs(helper)
    precise = "NAME-MATCH" not in out and "reference" in out
    ok = names[5] in out
    return {"tokens": _estimate_tokens(out), "calls": 1, "precise": precise, "ok": ok}


def _task_navigate(rm, names):
    helper, _collider, run, _container, lib, app = names
    t1 = rm.find_symbol(helper)
    t2 = rm.view_symbol(run)
    ok = lib in t1 and t2.startswith(f"{app}:")
    return {"tokens": _estimate_tokens(t1) + _estimate_tokens(t2), "calls": 2, "ok": ok}


def measure(lang):
    names = LANGS[lang]
    rm = RepoMap(os.path.join(FIXTURES, lang))
    rm._disk_checked = True  # hermetic: never touch the user's real tags cache
    t0 = time.monotonic()
    row = {"lang": lang,
           "qualified": _task_qualified(rm, names),
           "refs": _task_refs(rm, names),
           "navigate": _task_navigate(rm, names)}
    row["wall_ms"] = int((time.monotonic() - t0) * 1000)
    row["tokens_total"] = sum(row[t]["tokens"] for t in ("qualified", "refs", "navigate"))
    return row


class _NoLsp:
    """Hermetic stand-in: the metric accounts tool-output tokens, and must not
    spawn real language servers (network, uvx/npx fetches, per-machine variance).
    The `precise?` column therefore reports the no-server fallback labeling; the
    live precision path is validated separately by tests/test_lsp_live.py."""

    def references(self, *a, **k):
        return None

    def references_decorative(self, *a, **k):
        return None


def main():
    from chad import lsp
    lsp.service = lambda: _NoLsp()
    rows = [measure(lang) for lang in sorted(LANGS)]
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=2))
        return
    print(f"{'lang':<12}{'tok(qual)':>10}{'calls':>6}{'qual?':>7}{'tok(refs)':>10}"
          f"{'precise?':>9}{'tok(nav)':>9}{'total':>7}{'ms':>5}  all-ok")
    for r in rows:
        q, f, n = r["qualified"], r["refs"], r["navigate"]
        ok = all(x["ok"] for x in (q, f, n))
        print(f"{r['lang']:<12}{q['tokens']:>10}{q['calls']:>6}"
              f"{str(q['qualified_resolves']):>7}{f['tokens']:>10}"
              f"{str(f['precise']):>9}{n['tokens']:>9}{r['tokens_total']:>7}"
              f"{r['wall_ms']:>5}  {ok}")


if __name__ == "__main__":
    main()
