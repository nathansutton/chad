"""Pin lsp.py's degradation contract and operational guardrails.

The precision layer ships in the optional `lsp` extra (serena-agent), so nothing
here may require a live language server or the solidlsp package: extension routing,
graceful absence, the clangd compile-database gate, the decorative never-spawn
policy, and RSS-based recycling are all testable with fakes.
"""

import sys

from chad import lsp
from chad.lsp import LspService, lang_for


class _FakeServer:
    """Stands in for a running SolidLanguageServer."""

    def __init__(self, locs=()):
        self.locs = list(locs)
        self.stopped = False
        self.requests = 0

    def request_references(self, rel, row, col):
        self.requests += 1
        return self.locs

    def stop(self):
        self.stopped = True


def test_lang_for_known_and_unknown():
    assert lang_for("x.py") == "PYTHON"
    assert lang_for("sub/dir/x.PY") == "PYTHON"  # extension match is case-insensitive
    assert lang_for("x.unknownext") is None
    assert lang_for("no_extension") is None


def test_degrades_without_solidlsp(tmp_path, monkeypatch):
    # None in sys.modules makes `from solidlsp import ...` raise ImportError,
    # exactly like a default (no-extra) install where the package is absent.
    monkeypatch.setitem(sys.modules, "solidlsp", None)
    svc = LspService(str(tmp_path))
    assert svc.available("f.py") is False
    assert svc.references("f.py", 0, 0) is None
    assert svc.definition("f.py", 0, 0) is None
    svc.stop()  # no server was created; must be a no-op, not an error


def test_cpp_gated_on_compile_db(tmp_path, monkeypatch):
    """Without a compilation database clangd returns confidently-empty cross-file
    results, so the server must not start at all (callers get the honestly-labeled
    tree-sitter fallback). The gate sits before the solidlsp import, so it holds on
    a default install too."""
    monkeypatch.setitem(sys.modules, "solidlsp", None)
    svc = LspService(str(tmp_path))
    assert svc.references("main.cpp", 0, 0) is None
    assert svc._servers.get("CPP", "untried") is None  # cached refusal, no retry

    # with a compile db the gate opens (and then absence of solidlsp degrades as usual)
    (tmp_path / "compile_commands.json").write_text("[]")
    svc2 = LspService(str(tmp_path))
    assert svc2.references("main.cpp", 0, 0) is None  # solidlsp absent, not gated
    svc.stop()
    svc2.stop()


def test_decorative_never_spawns_foreign_server(tmp_path):
    """references_decorative must not pay a server start for decoration: an
    un-started non-Python language is refused without ever touching _server()."""
    svc = LspService(str(tmp_path))

    def boom(lang):
        if lang in svc._servers:  # cache hits are fine; *starting* one is the bug
            return svc._servers[lang]
        raise AssertionError(f"decorative call tried to start a {lang} server")

    svc._server = boom
    assert svc.references_decorative("x.rs", 0, 0) is None
    assert svc.references_decorative("x.ts", 0, 0) is None

    # but an already-running foreign server IS used
    fake = _FakeServer(locs=[{"relativePath": "user.rs",
                              "range": {"start": {"line": 3, "character": 4}}}])
    svc._servers["RUST"] = fake
    svc._recycle_if_bloated = lambda lang, srv: None
    assert svc.references_decorative("x.rs", 0, 0) == [("user.rs", 4, 5)]
    assert fake.requests == 1
    svc.stop()


def test_recycle_on_rss_cap(tmp_path, monkeypatch):
    """A server whose process tree exceeds the RSS cap is stopped and forgotten
    after the request, so the next call starts fresh instead of the bloated server
    starving the GPU allocator."""
    class _PidServer(_FakeServer):
        class server:  # noqa: N801 - mimic solidlsp's srv.server.process.pid shape
            class process:
                pid = 4242

    fat = _PidServer(locs=[])
    svc = LspService(str(tmp_path))
    svc._servers["PYTHON"] = fat
    monkeypatch.setattr(lsp, "_tree_rss_mb", lambda pid: lsp._MAX_RSS_MB + 1)
    assert svc.references("f.py", 0, 0) == []
    assert fat.stopped is True
    assert "PYTHON" not in svc._servers  # forgotten -> next request restarts

    # under the cap, the server is left alone
    slim = _PidServer(locs=[])
    svc._servers["PYTHON"] = slim
    monkeypatch.setattr(lsp, "_tree_rss_mb", lambda pid: 100)
    assert svc.references("f.py", 0, 0) == []
    assert slim.stopped is False
    assert svc._servers["PYTHON"] is slim
    svc.stop()
