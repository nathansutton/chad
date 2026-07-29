"""Pin lsp.py's degradation contract and operational guardrails.

The precision layer ships in the base install but its language servers live
outside the wheel (PATH / uvx / npx), so nothing here may require a live server:
extension routing, graceful absence, the honestly-unsupported rows, the clangd
compile-database gate, the decorative never-spawn policy, dead-server drop, and
RSS-based recycling are all testable with fakes. Protocol mechanics live in
test_lspclient.py; a real pyright run lives in test_lsp_live.py.
"""

import os

from chad import lsp, lspservers
from chad.lsp import LspService
from chad.lspservers import lang_for


class _FakeClient:
    """Stands in for a running lspclient.LspClient."""

    def __init__(self, locs=(), alive=True):
        self.locs = list(locs)
        self.stopped = False
        self.requests = 0
        self._alive = alive

    def sync_doc(self, path, language_id, text=None):
        return "file://" + path

    def references(self, path, row, col, timeout=5.0, include_declaration=False):
        self.requests += 1
        if not self._alive:
            raise RuntimeError("server gone")
        return self.locs

    def definition(self, path, row, col, timeout=5.0):
        return [(p, ln) for p, ln, _c in self.locs]

    def shutdown(self):
        self.stopped = True

    def alive(self):
        return self._alive

    @property
    def pid(self):
        return 4242


def test_lang_for_owns_its_identities():
    assert lang_for("x.py") == "python"
    assert lang_for("sub/dir/x.PY") == "python"  # extension match is case-insensitive
    # chad's own language identity, not a vendor enum: .js is javascript (the
    # typescript SERVER handles it, but the language is named for what it is),
    # and .c is c, not "CPP".
    assert lang_for("a.js") == "javascript"
    assert lang_for("a.c") == "c"
    assert lang_for("a.cpp") == "cpp"
    assert lang_for("x.unknownext") is None
    assert lang_for("no_extension") is None
    # languages that share a server share a process key
    assert lspservers.spec_for("javascript").server_key \
        == lspservers.spec_for("typescript").server_key
    assert lspservers.spec_for("c").server_key == lspservers.spec_for("cpp").server_key


def test_degrades_without_any_server(tmp_path, monkeypatch):
    # No resolvable command == a machine with no server installed: every request
    # degrades to None (tree-sitter fallback), nothing raises, stop() is a no-op.
    monkeypatch.setattr(lspservers, "resolve_commands", lambda spec: [])
    svc = LspService(str(tmp_path))
    assert svc.available("f.py") is False
    assert svc.references("f.py", 0, 0) is None
    assert svc.definition("f.py", 0, 0) is None
    assert svc.hover("f.py", 0, 0) is None
    assert svc.diagnostics_after_edit("f.py") is None
    svc.stop()  # no server was created; must be a no-op, not an error


def test_unsupported_language_refuses_honestly(tmp_path):
    """java/csharp rows declare themselves unsupported (empty commands) — the
    registry SAYS so instead of half-supporting a bootstrap chad doesn't own."""
    svc = LspService(str(tmp_path))
    assert svc.references("App.java", 0, 0) is None
    assert svc._servers.get("jdtls", "untried") is None  # cached refusal, no retry
    spec = lspservers.spec_for("java")
    assert spec.commands == () and "jdtls" in spec.unsupported
    svc.stop()


def test_cpp_gated_on_compile_db(tmp_path, monkeypatch):
    """Without a compilation database clangd returns confidently-empty cross-file
    results, so the server must not start at all (callers get the honestly-labeled
    tree-sitter fallback)."""
    svc = LspService(str(tmp_path))
    assert svc.references("main.cpp", 0, 0) is None
    assert svc._servers.get("clangd", "untried") is None  # cached refusal, no retry

    # with a compile db the gate opens (and an unresolvable command degrades as usual)
    (tmp_path / "compile_commands.json").write_text("[]")
    monkeypatch.setattr(lspservers, "resolve_commands", lambda spec: [])
    svc2 = LspService(str(tmp_path))
    assert svc2.references("main.cpp", 0, 0) is None  # command absent, not gated
    svc.stop()
    svc2.stop()


def test_typescript_gated_on_project_file(tmp_path, monkeypatch):
    """Same honesty rule as clangd: TS without a tsconfig/jsconfig returns
    confidently-empty cross-file results (measured on the TS7 native LSP), so the
    server must not start at all."""
    svc = LspService(str(tmp_path))
    assert svc.references("app.ts", 0, 0) is None
    assert svc._servers.get("typescript", "untried") is None
    (tmp_path / "tsconfig.json").write_text("{}")
    monkeypatch.setattr(lspservers, "resolve_commands", lambda spec: [])
    svc2 = LspService(str(tmp_path))
    assert svc2.references("app.ts", 0, 0) is None    # command absent, not gated
    svc.stop()
    svc2.stop()


def test_decorative_never_spawns_expensive_server(tmp_path):
    """references_decorative must not pay a server start for decoration unless the
    registry marks the server cheap (decorative_ok): an un-started rust/ts server
    is refused without ever touching _server()."""
    svc = LspService(str(tmp_path))

    def boom(spec):
        if spec.server_key in svc._servers:  # cache hits are fine; *starting* is the bug
            return svc._servers[spec.server_key]
        raise AssertionError(f"decorative call tried to start {spec.server_key}")

    svc._server = boom
    assert svc.references_decorative("x.rs", 0, 0) is None
    assert svc.references_decorative("x.ts", 0, 0) is None

    # but an already-running foreign server IS used
    fake = _FakeClient(locs=[(os.path.join(str(tmp_path), "user.rs"), 4, 5)])
    svc._servers["rust-analyzer"] = fake
    svc._recycle_if_bloated = lambda key, cli: None
    assert svc.references_decorative("x.rs", 0, 0) == [("user.rs", 4, 5)]
    assert fake.requests == 1
    svc.stop()


def test_edit_diagnostics_never_spawn_a_server(tmp_path):
    """An edit result is on the interactive path: diagnostics_after_edit must not
    start ANY server — not even the registry-cheap python one. (A cold uvx fetch
    inside an edit round trip, or inside every edit test of the fast gate, is the
    stall this pins out.)"""
    svc = LspService(str(tmp_path))

    def boom(spec):
        raise AssertionError(f"edit diagnostics tried to start {spec.server_key}")

    svc._server = boom
    assert svc.diagnostics_after_edit("f.py") is None
    assert svc.diagnostics_after_edit("app.ts") is None
    svc.stop()


def test_python_is_decorative_ok():
    """pyright is the one server proven cheap enough to spawn for a hint (~0.9s,
    flat across repo sizes) — the registry must keep saying so."""
    assert lspservers.spec_for("python").decorative_ok is True
    assert lspservers.spec_for("cpp").decorative_ok is False
    assert lspservers.spec_for("rust").decorative_ok is False


def test_recycle_on_rss_cap(tmp_path, monkeypatch):
    """A server whose process tree exceeds the RSS cap is stopped and forgotten
    after the request, so the next call starts fresh instead of the bloated server
    starving the GPU allocator."""
    fat = _FakeClient(locs=[])
    svc = LspService(str(tmp_path))
    svc._servers["pyright"] = fat
    monkeypatch.setattr(lsp, "_tree_rss_mb", lambda pid: lsp._MAX_RSS_MB + 1)
    assert svc.references("f.py", 0, 0) == []
    assert fat.stopped is True
    assert "pyright" not in svc._servers  # forgotten -> next request restarts

    # under the cap, the server is left alone
    slim = _FakeClient(locs=[])
    svc._servers["pyright"] = slim
    monkeypatch.setattr(lsp, "_tree_rss_mb", lambda pid: 100)
    assert svc.references("f.py", 0, 0) == []
    assert slim.stopped is False
    assert svc._servers["pyright"] is slim
    svc.stop()


def test_dead_server_is_dropped(tmp_path, monkeypatch):
    """A request against a dead server degrades to None AND forgets the server, so
    the next request can start a fresh one instead of failing forever."""
    dead = _FakeClient(alive=False)
    svc = LspService(str(tmp_path))
    svc._servers["pyright"] = dead
    monkeypatch.setattr(lsp, "_tree_rss_mb", lambda pid: 0)
    assert svc.references("f.py", 0, 0) is None
    assert "pyright" not in svc._servers
    svc.stop()


class _FakeDiagService:
    def __init__(self, diags):
        self._diags = diags
        self.root = os.path.abspath(os.getcwd())  # so service() won't rebind

    def diagnostics_after_edit(self, path):
        return self._diags


def _with_service(monkeypatch, svc):
    monkeypatch.setattr(lsp, "_SERVICE", svc)


def test_diagnostics_note_formats_errors_only(monkeypatch):
    _with_service(monkeypatch, _FakeDiagService(
        [("error", 3, "Type \"str\" is not assignable to \"int\"\nlong detail"),
         ("warning", 9, "unused variable"),
         ("error", 7, "x" * 500)]))
    note = lsp.diagnostics_note("f.py")
    assert "2 error(s)" in note, note
    assert "line 3: Type" in note and "line 7: xxx" in note
    assert "unused variable" not in note          # warnings are edit-time noise
    assert "long detail" not in note              # first line of each message only
    assert len(note) < 500                        # message truncation held


def test_diagnostics_note_caps_at_five(monkeypatch):
    _with_service(monkeypatch, _FakeDiagService(
        [("error", i, f"e{i}") for i in range(1, 9)]))
    note = lsp.diagnostics_note("f.py")
    assert "8 error(s)" in note and "(+3 more)" in note, note
    assert "line 5:" in note and "line 6:" not in note


def test_diagnostics_note_silent_paths(monkeypatch):
    # no warm server / clean bill / warnings only / disabled -> all empty
    _with_service(monkeypatch, _FakeDiagService(None))
    assert lsp.diagnostics_note("f.py") == ""
    _with_service(monkeypatch, _FakeDiagService([]))
    assert lsp.diagnostics_note("f.py") == ""
    _with_service(monkeypatch, _FakeDiagService([("warning", 1, "meh")]))
    assert lsp.diagnostics_note("f.py") == ""
    _with_service(monkeypatch, _FakeDiagService([("error", 1, "real")]))
    monkeypatch.setattr(lsp, "_DIAG_TIMEOUT_S", 0.0)
    assert lsp.diagnostics_note("f.py") == ""     # CHAD_LSP_DIAG_TIMEOUT=0 kill switch


def test_post_edit_diagnostics_reach_symbol_edits(tmp_path, monkeypatch):
    """The lever wires the typecheck note into replace_symbol results (and stays
    out when disabled)."""
    from chad.symbols import SymbolService
    fp = tmp_path / "m.py"
    fp.write_text("def f():\n    return 1\n")
    monkeypatch.setattr(lsp, "diagnostics_note",
                        lambda path: "\n[typecheck — 1 error(s) (language server):"
                                     "\n  line 2: boom]")
    svc = SymbolService(str(tmp_path))
    res = svc.replace_symbol("f", "def f():\n    return 2", path=str(fp))
    assert res.startswith("[replaced") and "typecheck" in res, res

    monkeypatch.setenv("CHAD_DISABLE", "post_edit_diagnostics")  # read per call
    fp2 = tmp_path / "n.py"
    fp2.write_text("def g():\n    return 1\n")
    res2 = svc.replace_symbol("g", "def g():\n    return 3", path=str(fp2))
    assert "typecheck" not in res2, res2


def test_hover_tool_routes_by_symbol_name(tmp_path, monkeypatch):
    """repomap.hover: name -> definition position -> language-server hover text;
    honest fallback when no server speaks for the language."""
    from chad.repomap import RepoMap
    (tmp_path / "m.py").write_text("def helper(x):\n    return x\n")
    rm = RepoMap(str(tmp_path))
    rm._disk_checked = True

    class _HoverSvc:
        def __init__(self):
            self.root = os.path.abspath(os.getcwd())
            self.asked = None

        def hover(self, rel, row, col):
            self.asked = (rel, row, col)
            return "(function) def helper(x: int) -> int"

    svc = _HoverSvc()
    _with_service(monkeypatch, svc)
    out = rm.hover("helper")
    assert out.startswith("m.py:1"), out
    assert "helper(x: int)" in out
    assert svc.asked == ("m.py", 0, 4)   # the identifier's 0-based position

    class _NoSvc:
        root = os.path.abspath(os.getcwd())

        def hover(self, rel, row, col):
            return None

    _with_service(monkeypatch, _NoSvc())
    assert "no language server available" in rm.hover("helper")
    assert "symbol not found" in rm.hover("nope")


def test_rel_paths_in_results(tmp_path, monkeypatch):
    """Absolute paths from the client come back repo-relative, matching what the
    tree-sitter fallback prints (the model sees one path style)."""
    fake = _FakeClient(locs=[(os.path.join(str(tmp_path), "pkg", "user.py"), 3, 7)])
    svc = LspService(str(tmp_path))
    svc._servers["pyright"] = fake
    monkeypatch.setattr(lsp, "_tree_rss_mb", lambda pid: 0)
    assert svc.references("f.py", 0, 0) == [(os.path.join("pkg", "user.py"), 3, 7)]
    svc.stop()
