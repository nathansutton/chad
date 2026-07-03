"""Precise cross-file code intelligence via an in-process language server.

Phase 2 of the symbolic stack. Tree-sitter (`repomap.py`) gives fast, language-
agnostic STRUCTURE; this adds semantic PRECISION that name-matching cannot:
true go-to-definition resolution and cross-file find-all-references that follow
imports, respect scoping, and don't confuse a method with an unrelated function of
the same name.

We drive `solidlsp` (Serena's synchronous LSP core) directly — no MCP server, no
Serena agent. For Python it spawns pyright on a worker thread (≈0.3s warm; a
one-time pyright download on a cold machine). The server is started LAZILY on first
use (so sessions that never ask for references pay nothing), bound to one project
root, and reaped on rebind / process exit.

Everything degrades gracefully: if the language server can't start (no Node,
offline, unsupported language) callers fall back to the tree-sitter backend. The
LSP is a precision upgrade, never a hard dependency of the harness.
"""

import atexit
import os
import threading

from .diag import log

# solidlsp pulls a fair bit of machinery; import lazily inside _server() so a normal
# session / cold start doesn't pay for it unless references are actually requested.
# It ships in the optional `lsp` extra (serena-agent) — absent, we log once and every
# caller degrades to the tree-sitter backend.

_MISSING_LOGGED = False

_IGNORED = ["__pycache__", ".git", ".venv", "venv", "node_modules", ".mypy_cache",
            ".pytest_cache", "dist", "build", ".cache", "models"]

# Extension -> solidlsp Language enum name. Python is the proven backend (pyright);
# the others light up automatically once their language server is available.
_LANG_BY_EXT = {
    ".py": "PYTHON", ".pyi": "PYTHON",
    ".ts": "TYPESCRIPT", ".tsx": "TYPESCRIPT", ".js": "TYPESCRIPT", ".jsx": "TYPESCRIPT",
    ".go": "GO", ".rs": "RUST", ".java": "JAVA", ".rb": "RUBY",
    ".cs": "CSHARP", ".php": "PHP", ".kt": "KOTLIN", ".cpp": "CPP", ".cc": "CPP",
    ".c": "CPP", ".h": "CPP", ".hpp": "CPP",
}


def lang_for(path: str):
    return _LANG_BY_EXT.get(os.path.splitext(path)[1].lower())


class LspService:
    """One language server per (root, language), started lazily, reaped on exit."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._servers = {}            # lang -> SolidLanguageServer | None (None = tried & failed)
        self._lock = threading.Lock()

    def _server(self, lang: str):
        """Lazily create + start the language server for `lang`; None if unavailable.
        A failed start is remembered (cached None) so we don't retry on every call."""
        if lang in self._servers:
            return self._servers[lang]
        with self._lock:
            if lang in self._servers:
                return self._servers[lang]
            srv = None
            try:
                from solidlsp import SolidLanguageServer
                from solidlsp.ls_config import Language, LanguageServerConfig
                from solidlsp.settings import SolidLSPSettings
            except ImportError:
                global _MISSING_LOGGED
                if not _MISSING_LOGGED:
                    _MISSING_LOGGED = True
                    log.info("lsp: solidlsp not installed (install the 'lsp' extra for "
                             "precise refs/rename); using tree-sitter fallback")
                self._servers[lang] = None
                return None
            try:
                cfg = LanguageServerConfig(code_language=Language[lang],
                                           ignored_paths=list(_IGNORED))
                srv = SolidLanguageServer.create(cfg, self.root,
                                                 solidlsp_settings=SolidLSPSettings())
                srv.start()
            except Exception:
                srv = None
            self._servers[lang] = srv
            return srv

    def stop(self):
        for srv in self._servers.values():
            if srv is not None:
                try:
                    srv.stop()
                except Exception:
                    pass
        self._servers.clear()

    def available(self, path: str) -> bool:
        lang = lang_for(path)
        return bool(lang) and self._server(lang) is not None

    def references(self, rel_path: str, name_row: int, name_col: int):
        """Precise references to the symbol whose identifier sits at the 0-based
        (name_row, name_col) in rel_path. Returns a list of (rel, line1, col1) or
        None if the language server is unavailable / errored (caller should fall
        back to tree-sitter). An empty list means "ran, found none"."""
        lang = lang_for(rel_path)
        if not lang:
            return None
        srv = self._server(lang)
        if srv is None:
            return None
        try:
            locs = srv.request_references(rel_path, name_row, name_col)
        except Exception:
            return None
        out = []
        for loc in locs or []:
            rel = loc.get("relativePath")
            if not rel:
                uri = loc.get("uri", "")
                rel = self._rel(uri[7:]) if uri.startswith("file://") else uri
            start = loc.get("range", {}).get("start", {})
            out.append((rel, start.get("line", 0) + 1, start.get("character", 0) + 1))
        return out

    def definition(self, rel_path: str, row: int, col: int):
        """Precise go-to-definition for the symbol at 0-based (row, col). Returns a
        list of (rel, line1) or None if unavailable."""
        lang = lang_for(rel_path)
        if not lang:
            return None
        srv = self._server(lang)
        if srv is None:
            return None
        try:
            locs = srv.request_definition(rel_path, row, col)
        except Exception:
            return None
        out = []
        for loc in locs or []:
            rel = loc.get("relativePath") or loc.get("uri", "")
            start = loc.get("range", {}).get("start", {})
            out.append((rel, start.get("line", 0) + 1))
        return out

    def _rel(self, p):
        try:
            return os.path.relpath(p, self.root)
        except ValueError:
            return p


_SERVICE = None


def service() -> LspService:
    """Lazily-bound, cwd-rooted service. Rebinding to a new project root (e.g. each
    eval task) stops the previous language server so pyright processes don't leak."""
    global _SERVICE
    cwd = os.path.abspath(os.getcwd())
    if _SERVICE is None or _SERVICE.root != cwd:
        if _SERVICE is not None:
            _SERVICE.stop()
        _SERVICE = LspService(cwd)
    return _SERVICE


@atexit.register
def _shutdown():
    if _SERVICE is not None:
        _SERVICE.stop()
