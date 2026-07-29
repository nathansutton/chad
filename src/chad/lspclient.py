"""Generic LSP client: Content-Length JSON-RPC over stdio. Zero language knowledge.

This is chad's own precision transport — the six-method surface we consumed from
solidlsp (spawn, initialize, references, definition, timeout, shutdown) plus the
two requests the context engine wants next (hover, publishDiagnostics), in one
file with no dependencies beyond the standard library. Which server to spawn for
which language, and how to tell when it is actually ready, is DATA and lives in
`lspservers.py`; policy (timeouts, RSS recycling, decorative spawns, fallback to
tree-sitter) stays in `lsp.py`.

Wire mechanics, all per the LSP spec:
* frames are `Content-Length: N\\r\\n\\r\\n` + N bytes of JSON, both directions;
* a reader thread correlates responses to requests by id (each request gets an
  Event; timeouts abandon the slot so a late response is dropped silently);
* server→client REQUESTS (workspace/configuration, client/registerCapability,
  window/workDoneProgress/create) must be answered or servers stall — we answer
  neutrally (configuration gets [null]×items, the rest get null);
* notifications we care about are recorded: window/logMessage into a ring buffer
  (readiness probes grep it — see lspservers' pyright rule) and
  textDocument/publishDiagnostics into a per-uri generation counter + payload so
  a caller can wait for the diagnostics that follow ITS edit, not a stale set.
"""

import itertools
import json
import os
import pathlib
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque

from .diag import log


class LspError(Exception):
    """A request the server answered with an error, or a dead/unusable server."""


def path_to_uri(path: str) -> str:
    return pathlib.Path(os.path.abspath(path)).as_uri()


def uri_to_path(uri: str) -> str:
    return urllib.request.url2pathname(urllib.parse.urlparse(uri).path)


class LspClient:
    """One language-server process: spawn, handshake, request/notify, shutdown."""

    def __init__(self, cmd, root: str):
        self.root = os.path.abspath(root)
        self.cmd = list(cmd)
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
        self._seq = itertools.count(1)
        self._write_lock = threading.Lock()
        self._pending = {}          # id -> (Event, slot dict)
        self._pending_lock = threading.Lock()
        self._cond = threading.Condition()   # guards _logs and _diags updates
        self._logs = deque(maxlen=256)       # window/logMessage message strings
        self._diags = {}            # uri -> (generation, [diagnostic dicts])
        self._open_docs = {}        # uri -> version
        self._reader = threading.Thread(target=self._read_loop, daemon=True,
                                        name=f"lsp-read-{os.path.basename(self.cmd[0])}")
        self._reader.start()

    # -- wire --------------------------------------------------------------

    def _send(self, payload: dict):
        body = json.dumps(payload).encode()
        frame = b"Content-Length: %d\r\n\r\n%s" % (len(body), body)
        stdin = self.proc.stdin
        try:
            with self._write_lock:
                assert stdin is not None   # Popen(stdin=PIPE) guarantees it
                stdin.write(frame)
                stdin.flush()
        except Exception as e:
            raise LspError(f"server unwritable: {e}") from e

    def _read_loop(self):
        stdout = self.proc.stdout
        try:
            while True:
                length = None
                while True:                    # headers until the blank line
                    line = stdout.readline()
                    if not line:
                        raise EOFError
                    if line in (b"\r\n", b"\n"):
                        break
                    if line.lower().startswith(b"content-length:"):
                        length = int(line.split(b":", 1)[1].strip())
                if length is None:
                    continue
                body = stdout.read(length)
                if body is None or len(body) < length:
                    raise EOFError
                try:
                    self._dispatch(json.loads(body))
                except Exception:  # noqa: BLE001 - one bad frame must not kill the reader
                    log.info("lsp: undispatchable frame from %s", self.cmd[0])
        except Exception:  # noqa: BLE001 - EOF/kill: fail everything still waiting
            with self._pending_lock:
                pending, self._pending = self._pending, {}
            for ev, slot in pending.values():
                slot["error"] = {"message": "server exited"}
                ev.set()

    def _dispatch(self, msg: dict):
        if "id" in msg and "method" in msg:      # server -> client request
            self._answer_server_request(msg)
        elif "method" in msg:                    # notification
            params = msg.get("params") or {}
            if msg["method"] == "window/logMessage":
                with self._cond:
                    self._logs.append(str(params.get("message", "")))
                    self._cond.notify_all()
            elif msg["method"] == "textDocument/publishDiagnostics":
                uri = params.get("uri", "")
                with self._cond:
                    gen = self._diags.get(uri, (0, None))[0] + 1
                    self._diags[uri] = (gen, params.get("diagnostics") or [])
                    self._cond.notify_all()
        elif "id" in msg:                        # response
            with self._pending_lock:
                ev_slot = self._pending.pop(msg["id"], None)
            if ev_slot:                          # None: timed out, dropped
                ev, slot = ev_slot
                if "error" in msg:
                    slot["error"] = msg["error"]
                else:
                    slot["result"] = msg.get("result")
                ev.set()

    def _answer_server_request(self, msg: dict):
        # A server blocked on an unanswered request looks exactly like a hang.
        result = None
        if msg["method"] == "workspace/configuration":
            items = (msg.get("params") or {}).get("items") or []
            result = [None] * len(items)
        try:
            self._send({"jsonrpc": "2.0", "id": msg["id"], "result": result})
        except LspError:
            pass

    # -- rpc ---------------------------------------------------------------

    def request(self, method: str, params=None, timeout: float = 5.0):
        rid = next(self._seq)
        ev, slot = threading.Event(), {}
        with self._pending_lock:
            self._pending[rid] = (ev, slot)
        self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                    "params": params or {}})
        if not ev.wait(timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"{method} timed out after {timeout:g}s")
        if "error" in slot:
            raise LspError(f"{method}: {slot['error'].get('message', slot['error'])}")
        return slot.get("result")

    def notify(self, method: str, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    @property
    def pid(self):
        return self.proc.pid

    def alive(self) -> bool:
        return self.proc.poll() is None

    # -- lifecycle ---------------------------------------------------------

    def initialize(self, init_options=None, timeout: float = 30.0) -> dict:
        uri = path_to_uri(self.root)
        result = self.request("initialize", {
            "processId": os.getpid(),
            "rootUri": uri,
            "rootPath": self.root,
            "workspaceFolders": [{"uri": uri, "name": os.path.basename(self.root)}],
            "initializationOptions": init_options or {},
            "capabilities": {
                "textDocument": {
                    "synchronization": {"didSave": True},
                    "references": {}, "definition": {},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "publishDiagnostics": {},
                    "diagnostic": {},   # pull model — TS7's native LSP never pushes
                },
                "workspace": {"workspaceFolders": True, "configuration": True},
            },
        }, timeout=timeout)
        self.notify("initialized", {})
        return result or {}

    def shutdown(self):
        try:
            self.request("shutdown", timeout=2.0)
        except Exception:  # noqa: BLE001 - a busy server still gets exit+kill below
            pass
        try:
            self.notify("exit")
        except Exception:  # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=1.0)
        except Exception:  # noqa: BLE001
            self.proc.kill()

    # -- documents ---------------------------------------------------------

    def sync_doc(self, path: str, language_id: str, text=None) -> str:
        """didOpen (first time) or full-text didChange (after) for `path`; returns
        its uri. `text=None` reads the file — pass text to sync unsaved content.
        Unchanged content is NOT re-sent (an agent re-reads hot files constantly;
        redundant didChange floods make servers re-analyze for nothing)."""
        if text is None:
            with open(path, errors="replace") as f:
                text = f.read()
        uri = path_to_uri(path)
        ver_hash = self._open_docs.get(uri)
        if ver_hash is None:
            self._open_docs[uri] = (1, hash(text))
            self.notify("textDocument/didOpen", {"textDocument": {
                "uri": uri, "languageId": language_id, "version": 1, "text": text}})
        elif ver_hash[1] != hash(text):
            ver = ver_hash[0] + 1
            self._open_docs[uri] = (ver, hash(text))
            self.notify("textDocument/didChange", {
                "textDocument": {"uri": uri, "version": ver},
                "contentChanges": [{"text": text}]})
        return uri

    # -- requests the context engine uses ---------------------------------

    def references(self, path: str, row: int, col: int, timeout: float = 5.0,
                   include_declaration: bool = False):
        """[(abs_path, line1, col1)] for the symbol at 0-based (row, col)."""
        uri = path_to_uri(path)
        locs = self.request("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": row, "character": col},
            "context": {"includeDeclaration": include_declaration},
        }, timeout=timeout)
        return [self._loc(loc) for loc in locs or []]

    def definition(self, path: str, row: int, col: int, timeout: float = 5.0):
        """[(abs_path, line1)] — Location | Location[] | LocationLink[] normalized."""
        res = self.request("textDocument/definition", {
            "textDocument": {"uri": path_to_uri(path)},
            "position": {"line": row, "character": col},
        }, timeout=timeout)
        if isinstance(res, dict):
            res = [res]
        return [self._loc(loc)[:2] for loc in res or []]

    def hover(self, path: str, row: int, col: int, timeout: float = 5.0):
        """Plain-text hover contents for (row, col), or None."""
        res = self.request("textDocument/hover", {
            "textDocument": {"uri": path_to_uri(path)},
            "position": {"line": row, "character": col},
        }, timeout=timeout)
        if not res:
            return None
        return _hover_text(res.get("contents")) or None

    @staticmethod
    def _loc(loc: dict):
        uri = loc.get("uri") or loc.get("targetUri", "")
        rng = loc.get("range") or loc.get("targetSelectionRange") or {}
        start = rng.get("start") or {}
        return (uri_to_path(uri), start.get("line", 0) + 1,
                start.get("character", 0) + 1)

    def pull_diagnostics(self, path: str, timeout: float = 5.0):
        """textDocument/diagnostic (LSP 3.17 pull model): the server's current
        diagnostics for `path`, synchronously. Raises on servers that don't
        implement it (callers then fall back to the push stream)."""
        res = self.request("textDocument/diagnostic", {
            "textDocument": {"uri": path_to_uri(path)},
        }, timeout=timeout)
        if isinstance(res, dict):
            return res.get("items") or []
        return []

    # -- push streams ------------------------------------------------------

    def diag_generation(self, path: str) -> int:
        with self._cond:
            return self._diags.get(path_to_uri(path), (0, None))[0]

    def wait_diagnostics(self, path: str, after_generation: int, timeout: float = 5.0):
        """The first publishDiagnostics for `path` newer than `after_generation`
        (i.e. the analysis of OUR edit, not a stale set), or None on timeout."""
        uri = path_to_uri(path)
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                gen, diags = self._diags.get(uri, (0, None))
                if gen > after_generation:
                    return diags
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cond.wait(remaining):
                    return None

    def wait_log(self, pattern: str, timeout: float) -> bool:
        """True once any window/logMessage (past or future) matches `pattern` —
        how registry readiness rules block until a server stops lying (pyright
        answers references with a confident [] until its scan logs completion)."""
        rx = re.compile(pattern)
        deadline = time.monotonic() + timeout
        scanned = 0
        with self._cond:
            while True:
                logs = list(self._logs)
                if any(rx.search(m) for m in logs[scanned:]):
                    return True
                scanned = len(logs)
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cond.wait(remaining):
                    return False


def _hover_text(contents) -> str:
    """Hover contents in all three spec shapes, flattened to plain text."""
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents.strip()
    if isinstance(contents, dict):     # MarkupContent | MarkedString {language,value}
        return str(contents.get("value", "")).strip()
    if isinstance(contents, list):
        return "\n".join(filter(None, (_hover_text(c) for c in contents))).strip()
    return str(contents).strip()
