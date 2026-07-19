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

Operational guardrails (all measured on an 11k-file repo before they existed):
* every request carries a timeout (default 5s; one uncapped find-references ran 18s),
* C/C++ needs a compile database before clangd is started at all — without one
  clangd returns clean *empty* results and "[no references]" would be a lie,
* a server whose process tree outgrows CHAD_LSP_MAX_RSS_MB is recycled after the
  request (pyright hit 4 GB on hot symbols — on a machine wiring ~12 GB of model
  weights that's the same Metal-OOM crash class the repo map fix removed), and
* decorative callers (disambiguation hints) use `references_decorative`, which
  never *starts* a server except the proven-cheap Python backend.
"""

import atexit
import os
import subprocess
import threading
from collections import defaultdict

from . import config
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


_REQUEST_TIMEOUT_S = config.env_float("CHAD_LSP_TIMEOUT", 5.0)
_MAX_RSS_MB = config.env_int("CHAD_LSP_MAX_RSS_MB", 1536)

# Languages worth a server *start* for a decorative annotation. Python's pyright is
# proven lazy and cheap (~0.9s spawn, flat across repo sizes); other servers can
# gradle-sync, download binaries, or chew a file for ~10s (measured: clangd), which
# is never worth a "used by …" hint.
_DECOR_SPAWN = frozenset({"PYTHON"})


def _has_compile_db(root: str) -> bool:
    """clangd is only precise with a compilation database; without one it silently
    analyzes each file in isolation and returns clean empty cross-file results."""
    return any(os.path.exists(os.path.join(root, p))
               for p in ("compile_commands.json",
                         os.path.join("build", "compile_commands.json"),
                         ".clangd"))


_QUIET_HOOK_INSTALLED = False


def _install_quiet_shutdown_hook():
    """solidlsp sends the LSP shutdown request from a helper thread; when the server
    is busy (typically mid-request after we timed one out) that request itself times
    out and the default threading excepthook dumps a full traceback over the TUI.
    Filter exactly that case — solidlsp still reaps the process — and pass every
    other thread exception through untouched."""
    global _QUIET_HOOK_INSTALLED
    if _QUIET_HOOK_INSTALLED:
        return
    _QUIET_HOOK_INSTALLED = True
    prev = threading.excepthook

    def hook(args):
        if (args.exc_type is TimeoutError
                and "_send_shutdown" in getattr(args.thread, "name", "")):
            log.info("lsp: shutdown request timed out (server busy); reaped anyway")
            return
        prev(args)

    threading.excepthook = hook


def _tree_rss_mb(pid: int) -> int:
    """Resident set of `pid` plus all descendants, in MB (pyright-langserver forks
    node children; the parent alone under-reports)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:  # noqa: BLE001 - a failed sample must never break a request
        return 0
    kids, rss = defaultdict(list), {}
    for ln in out.splitlines():
        try:
            p, pp, r = ln.split()
            kids[int(pp)].append(int(p))
            rss[int(p)] = int(r)
        except ValueError:
            continue
    total, stack, seen = 0, [pid], set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        total += rss.get(cur, 0)
        stack.extend(kids.get(cur, ()))
    return total // 1024


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
            if lang == "CPP" and not _has_compile_db(self.root):
                # Without a compile database clangd "works" but sees each file in
                # isolation: cross-file references come back a clean, WRONG empty.
                # Refusing to start keeps callers on the tree-sitter fallback, which
                # labels itself NAME-MATCH ONLY instead of lying with confidence.
                log.info("lsp: no compile_commands.json under %s; not starting clangd "
                         "(tree-sitter fallback)", self.root)
                self._servers[lang] = None
                return None
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
                _install_quiet_shutdown_hook()
                cfg = LanguageServerConfig(code_language=Language[lang],
                                           ignored_paths=list(_IGNORED))
                srv = SolidLanguageServer.create(cfg, self.root,
                                                 solidlsp_settings=SolidLSPSettings())
                srv.start()
                # A hung/slow request must degrade to the tree-sitter fallback, not
                # stall the agent loop (measured 18s for one uncapped find-references).
                try:
                    srv.set_request_timeout(_REQUEST_TIMEOUT_S)
                except Exception:  # noqa: BLE001 - older solidlsp without the knob
                    pass
            except Exception:
                srv = None
            self._servers[lang] = srv
            return srv

    def _recycle_if_bloated(self, lang: str, srv):
        """Stop and forget a server whose process tree outgrew the RSS cap. The next
        request simply starts a fresh one (pyright re-warms in ~2s) — unbounded
        analysis memory next to wired model weights is how the process dies."""
        pid = getattr(getattr(getattr(srv, "server", None), "process", None), "pid", None)
        if not pid:
            return
        mb = _tree_rss_mb(pid)
        if mb > _MAX_RSS_MB:
            log.info("lsp: %s server tree at %d MB (cap %d); recycling", lang, mb, _MAX_RSS_MB)
            try:
                srv.stop()
            except Exception:  # noqa: BLE001 - reaping is best-effort
                pass
            with self._lock:
                if self._servers.get(lang) is srv:
                    del self._servers[lang]

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

    def references(self, rel_path: str, name_row: int, name_col: int, timeout=None):
        """Precise references to the symbol whose identifier sits at the 0-based
        (name_row, name_col) in rel_path. Returns a list of (rel, line1, col1) or
        None if the language server is unavailable / errored / timed out (caller
        should fall back to tree-sitter). An empty list means "ran, found none".
        `timeout` tightens the request deadline below the default for callers with
        their own budget (the disambiguation pass); never loosens it."""
        lang = lang_for(rel_path)
        if not lang:
            return None
        srv = self._server(lang)
        if srv is None:
            return None
        set_to = getattr(srv, "set_request_timeout", None)
        if timeout is not None and set_to is not None:
            try:
                set_to(min(timeout, _REQUEST_TIMEOUT_S))
            except Exception:  # noqa: BLE001 - keep the default timeout
                set_to = None
        try:
            locs = srv.request_references(rel_path, name_row, name_col)
        except Exception:
            return None
        finally:
            if timeout is not None and set_to is not None:
                try:
                    set_to(_REQUEST_TIMEOUT_S)
                except Exception:  # noqa: BLE001
                    pass
            self._recycle_if_bloated(lang, srv)
        out = []
        for loc in locs or []:
            rel = loc.get("relativePath")
            if not rel:
                uri = loc.get("uri", "")
                rel = self._rel(uri[7:]) if uri.startswith("file://") else uri
            start = loc.get("range", {}).get("start", {})
            out.append((rel, start.get("line", 0) + 1, start.get("character", 0) + 1))
        return out

    def references_decorative(self, rel_path: str, name_row: int, name_col: int,
                              timeout=None):
        """`references()` for decoration (the disambiguation "used by …" hints):
        never pays a server *start* for it unless the language is the proven-cheap
        Python backend — other languages are used only if already running. Measured
        before this existed: one view_symbol('main') on a mixed repo spawned clangd
        per C++ candidate and took 86s."""
        lang = lang_for(rel_path)
        if not lang:
            return None
        if lang not in _DECOR_SPAWN and self._servers.get(lang) is None:
            return None  # absent (never started) and tried-but-failed both refuse
        return self.references(rel_path, name_row, name_col, timeout=timeout)

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
        finally:
            self._recycle_if_bloated(lang, srv)
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
