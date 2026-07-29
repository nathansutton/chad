"""Precise cross-file code intelligence via language servers chad drives itself.

Phase 2 of the symbolic stack. Tree-sitter (`repomap.py`) gives fast, language-
agnostic STRUCTURE; this adds semantic PRECISION that name-matching cannot:
true go-to-definition, cross-file find-all-references that follow imports and
respect scoping, hover types, and post-edit diagnostics.

The transport is chad's own `lspclient.py` (generic JSON-RPC, zero language
knowledge); which server speaks for which language is data in `lspservers.py`.
This module is POLICY: servers start LAZILY on first use (a session that never
asks for references pays nothing), are bound to one project root, degrade to the
tree-sitter fallback on any failure, and are reaped on rebind / process exit.
Precision ships in the base install — no extra, no vendored language server; a
missing server on PATH degrades per-language, never breaks the harness.

Operational guardrails (all measured on an 11k-file repo before they existed):
* every request carries a timeout (default 5s; one uncapped find-references ran 18s),
* C/C++ needs a compile database before clangd is started at all — without one
  clangd returns clean *empty* results and "[no references]" would be a lie,
* a server whose process tree outgrows CHAD_LSP_MAX_RSS_MB is recycled after the
  request (pyright hit 4 GB on hot symbols — on a machine wiring ~12 GB of model
  weights that's the same Metal-OOM crash class the repo map fix removed), and
* decorative callers (disambiguation hints) use `references_decorative`, which
  never *starts* a server unless the registry marks it `decorative_ok`
  (measured: one view_symbol('main') spawned clangd per candidate: 86s).
"""

import atexit
import os
import subprocess
import threading
from collections import defaultdict

from . import config, lspservers
from .diag import log
from .lspservers import lang_for  # re-export: the service keys requests off it

_REQUEST_TIMEOUT_S = config.env_float("CHAD_LSP_TIMEOUT", 5.0)
_INIT_TIMEOUT_S = config.env_float("CHAD_LSP_INIT_TIMEOUT", 30.0)
_DIAG_TIMEOUT_S = config.env_float("CHAD_LSP_DIAG_TIMEOUT", 3.0)
_MAX_RSS_MB = config.env_int("CHAD_LSP_MAX_RSS_MB", 1536)


def _gate_open(spec, root: str) -> bool:
    """The registry's honesty gate: some servers (clangd without a compile
    database, TS without a tsconfig — both measured) return confidently EMPTY
    cross-file results when the project context is missing. If the row names
    project files, one must exist or the server is not started at all."""
    return not spec.project_gate or any(
        os.path.exists(os.path.join(root, p)) for p in spec.project_gate)


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
    """One language-server process per (root, server_key) — servers that speak
    several languages (tsserver: ts+js; clangd: c+cpp) share one — started
    lazily, reaped on exit."""

    def __init__(self, root: str = "."):
        self.root = os.path.abspath(root)
        self._servers = {}    # server_key -> LspClient | None (None = tried & failed)
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def _server(self, spec):
        """Lazily create + start the language server for `spec`; None if
        unavailable. A failed start is remembered (cached None) — no per-call retry."""
        key = spec.server_key
        if key in self._servers:
            return self._servers[key]
        with self._lock:
            if key in self._servers:
                return self._servers[key]
            cli = None
            if not spec.commands:
                log.info("lsp: %s has no supported server (%s); tree-sitter fallback",
                         spec.language, spec.unsupported)
            elif not _gate_open(spec, self.root):
                # e.g. clangd without compile_commands.json, TS without a tsconfig:
                # the server would return clean, WRONG empties. Refusing keeps
                # callers on the tree-sitter fallback, which labels itself
                # NAME-MATCH ONLY instead of lying with confidence.
                log.info("lsp: none of %s under %s; not starting %s "
                         "(tree-sitter fallback)", spec.project_gate, self.root, key)
            else:
                cands = lspservers.resolve_commands(spec)
                if not cands:
                    log.info("lsp: no %s server available for %s (tier: %s); "
                             "tree-sitter fallback", key, spec.language, spec.tier)
                for cmd in cands:   # a candidate that fails to start must not
                    cli = self._start(spec, cmd)   # block the next one
                    if cli is not None:
                        break
            self._servers[key] = cli
            return cli

    def _start(self, spec, cmd):
        from . import lspclient
        cli = None
        try:
            cli = lspclient.LspClient(cmd, self.root)
            cli.initialize(init_options={}, timeout=_INIT_TIMEOUT_S)
            if spec.settings is not None:
                # pyright asks for workspace/configuration but does not START
                # analysis until settings are pushed (measured: refs hang forever
                # on a client that only ever answers the pull)
                cli.notify("workspace/didChangeConfiguration",
                           {"settings": spec.settings})
            if spec.ready_log:
                # e.g. pyright answers references with a confident EMPTY list until
                # its initial scan logs completion; asking earlier returns lies.
                cli.wait_log(spec.ready_log, timeout=spec.ready_timeout_s)
            return cli
        except Exception as e:  # noqa: BLE001 - any start failure degrades cleanly
            log.info("lsp: %s failed to start (%s); tree-sitter fallback",
                     spec.server_key, e)
            if cli is not None:
                try:
                    cli.shutdown()
                except Exception:  # noqa: BLE001
                    pass
            return None

    def _recycle_if_bloated(self, key, cli):
        """Stop and forget a server whose process tree outgrew the RSS cap. The next
        request simply starts a fresh one (pyright re-warms in ~2s) — unbounded
        analysis memory next to wired model weights is how the process dies."""
        mb = _tree_rss_mb(cli.pid) if cli.pid else 0
        if mb > _MAX_RSS_MB:
            log.info("lsp: %s server tree at %d MB (cap %d); recycling", key, mb,
                     _MAX_RSS_MB)
            self._drop(key, cli)

    def _drop(self, key, cli):
        try:
            cli.shutdown()
        except Exception:  # noqa: BLE001 - reaping is best-effort
            pass
        with self._lock:
            if self._servers.get(key) is cli:
                del self._servers[key]

    def stop(self):
        for cli in self._servers.values():
            if cli is not None:
                try:
                    cli.shutdown()
                except Exception:  # noqa: BLE001
                    pass
        self._servers.clear()

    def available(self, path: str) -> bool:
        spec = self._spec(path)
        return spec is not None and self._server(spec) is not None

    # -- requests ----------------------------------------------------------

    def _spec(self, rel_path):
        lang = lang_for(rel_path)
        return lspservers.spec_for(lang) if lang else None

    def _abs(self, rel_path):
        return rel_path if os.path.isabs(rel_path) else os.path.join(self.root, rel_path)

    def _rel(self, p):
        try:
            return os.path.relpath(p, self.root)
        except ValueError:
            return p

    def _ask(self, rel_path, fn, timeout=None, sync=True):
        """Common request plumbing: spec -> server -> didOpen -> fn(cli, abs, t),
        degrading to None on any failure; RSS check + dead-server drop after.
        `sync=False` leaves the didOpen/didChange to fn (diagnostics must read the
        publish generation BEFORE syncing or it can miss its own publish)."""
        spec = self._spec(rel_path)
        if spec is None:
            return None
        cli = self._server(spec)
        if cli is None:
            return None
        t = _REQUEST_TIMEOUT_S if timeout is None else min(timeout, _REQUEST_TIMEOUT_S)
        ap = self._abs(rel_path)
        try:
            if sync:
                cli.sync_doc(ap, spec.language_id)
            return fn(cli, ap, t)
        except Exception:  # noqa: BLE001 - timeout/protocol error/dead server
            if not cli.alive():
                self._drop(spec.server_key, cli)   # next request starts fresh
            return None
        finally:
            self._recycle_if_bloated(spec.server_key, cli)

    def references(self, rel_path: str, name_row: int, name_col: int, timeout=None):
        """Precise references to the symbol whose identifier sits at the 0-based
        (name_row, name_col) in rel_path. Returns a list of (rel, line1, col1) or
        None if the language server is unavailable / errored / timed out (caller
        should fall back to tree-sitter). An empty list means "ran, found none".
        `timeout` tightens the request deadline below the default for callers with
        their own budget (the disambiguation pass); never loosens it."""
        out = self._ask(rel_path,
                        lambda cli, ap, t: cli.references(ap, name_row, name_col,
                                                          timeout=t),
                        timeout)
        if out is None:
            return None
        return [(self._rel(p), ln, col) for p, ln, col in out]

    def references_decorative(self, rel_path: str, name_row: int, name_col: int,
                              timeout=None):
        """`references()` for decoration (the disambiguation "used by …" hints):
        never pays a server *start* unless the registry marks the server cheap
        (`decorative_ok`) — others are used only if already running."""
        spec = self._spec(rel_path)
        if spec is None:
            return None
        if not spec.decorative_ok and self._servers.get(spec.server_key) is None:
            return None  # absent (never started) and tried-but-failed both refuse
        return self.references(rel_path, name_row, name_col, timeout=timeout)

    def definition(self, rel_path: str, row: int, col: int):
        """Precise go-to-definition for the symbol at 0-based (row, col). Returns a
        list of (rel, line1) or None if unavailable."""
        out = self._ask(rel_path,
                        lambda cli, ap, t: cli.definition(ap, row, col, timeout=t))
        if out is None:
            return None
        return [(self._rel(p), ln) for p, ln in out]

    def hover(self, rel_path: str, row: int, col: int):
        """Hover text (type signature / docs) for the symbol at 0-based (row, col),
        or None if unavailable — the model reads a type in ~30 tokens instead of
        opening the defining file."""
        return self._ask(rel_path,
                         lambda cli, ap, t: cli.hover(ap, row, col, timeout=t))

    def diagnostics_after_edit(self, rel_path: str):
        """The server's diagnostics for rel_path's CURRENT on-disk content — the
        post-edit typecheck. Returns [(severity, line1, message)] (possibly empty:
        a clean bill), or None when no server is WARM for this language. Stricter
        than the decorative policy: an edit result is on the interactive path, so
        it never pays a server start at all — not even a registry-cheap one (a
        cold uvx fetch inside an edit round trip is exactly the stall the
        decorative rule exists to prevent). The note lights up once find_refs /
        hover / disambiguation has warmed the server."""
        spec = self._spec(rel_path)
        if spec is None:
            return None
        if self._servers.get(spec.server_key) is None:
            return None
        def ask(cli, ap, _t):
            gen = cli.diag_generation(ap)     # BEFORE the sync that triggers analysis
            cli.sync_doc(ap, spec.language_id)
            try:
                # pull first (LSP 3.17; synchronous, and unsupported servers refuse
                # instantly) — TS7's native LSP never pushes at all
                diags = cli.pull_diagnostics(ap, timeout=_DIAG_TIMEOUT_S)
            except Exception:  # noqa: BLE001 - no pull support: wait for the push
                diags = cli.wait_diagnostics(ap, gen, timeout=_DIAG_TIMEOUT_S)
            if diags is None:
                return None
            sev_names = {1: "error", 2: "warning", 3: "info", 4: "hint"}
            out = []
            for d in diags:
                start = (d.get("range") or {}).get("start") or {}
                out.append((sev_names.get(d.get("severity", 1), "error"),
                            start.get("line", 0) + 1, str(d.get("message", ""))))
            return out
        return self._ask(rel_path, ask, sync=False)


def diagnostics_note(path: str) -> str:
    """A compact post-edit typecheck note for edit-tool results, or "". Errors only
    (warnings are style noise mid-edit), capped at 5, first line of each message —
    ~50 tokens that can replace the failed test run the model would otherwise burn
    to discover a type error. Empty when no server is warm for the language (see
    diagnostics_after_edit's spawn policy) or diagnostics are disabled
    (CHAD_LSP_DIAG_TIMEOUT=0)."""
    if _DIAG_TIMEOUT_S <= 0:
        return ""
    try:
        diags = service().diagnostics_after_edit(path)
    except Exception:  # noqa: BLE001 - a diag failure must never break an edit result
        return ""
    errs = [(ln, msg) for sev, ln, msg in diags or [] if sev == "error"]
    if not errs:
        return ""
    lines = [f"  line {ln}: {msg.splitlines()[0][:160]}" for ln, msg in errs[:5]]
    more = f"\n  … (+{len(errs) - 5} more)" if len(errs) > 5 else ""
    return (f"\n[typecheck — {len(errs)} error(s) (language server):\n"
            + "\n".join(lines) + more + "]")


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
