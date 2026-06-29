"""OAuth client support for hosted HTTP MCP servers (plan 017).

Hosted connectors that matter to the operator (Linear, Slack, Atlassian, most SaaS)
require **OAuth**, not a static bearer token. Plan 016 added HTTP transport with
bearer/PAT auth; this module adds the OAuth path. It is built on the official `mcp`
SDK's OAuth machinery and contributes the two chad-specific pieces the SDK leaves to
the embedder:

  1. **Token persistence** — an `mcp.client.auth.TokenStorage` implementation that
     keeps per-server tokens + dynamic-client-registration info in
     `~/.chad/mcp_tokens.json`, created mode 0600 (never world-readable, never logged).
  2. **The interactive browser/loopback flow** — a one-shot localhost HTTP server that
     catches the `?code=...&state=...` redirect, plus a redirect handler that opens
     the browser (and prints the URL as a headless fallback). Bounded by a timeout so
     an abandoned or headless login degrades gracefully instead of hanging.

Everything here is gated by the `CHAD_MCP_OAUTH` feature flag (see `oauth_enabled`).
With the flag off, an `auth: oauth` server is skipped with a warning and the
stdio/bearer/HTTP paths from plan 016 are unchanged — none of this code runs.

SDK API confirmed against **mcp==1.28.1** (`mcp.__version__` does not exist; use
`importlib.metadata.version("mcp")`):
  - `from mcp.client.auth import OAuthClientProvider, TokenStorage`
  - `OAuthClientProvider(server_url, client_metadata, storage, redirect_handler,
     callback_handler, timeout=300.0, client_metadata_url=None)`
  - `TokenStorage` is a 4-method **async** protocol:
        get_tokens()       -> OAuthToken | None
        set_tokens(tokens) -> None
        get_client_info()  -> OAuthClientInformationFull | None
        set_client_info(client_info) -> None
  - models live in `mcp.shared.auth`: `OAuthClientMetadata` (only `redirect_uris`
    is required), `OAuthToken` (`access_token` required), `OAuthClientInformationFull`.
  - `OAuthClientProvider` is an `httpx.Auth`; Dynamic Client Registration is performed
    internally during its auth flow when the server advertises a registration endpoint
    and no client_info is stored (so we never hand-roll DCR).
  - `streamablehttp_client(url, headers=..., auth=<provider>)` accepts the provider —
    this is exactly why plan 016 keeps the deprecated `streamablehttp_client` (the
    newer `streamable_http_client` has no `auth=`).
No divergence from the plan's assumed API was found.

NOTE / e2e gap: a full 3-legged OAuth handshake against a real authorization server is
NOT exercised in tests (there is no real IdP in CI). The seams are tested — token-store
round-trip + 0600 perms, the loopback callback parsing a fired redirect, the flag-off
skip, the needs-login state, and headless/no-browser degradation. Do not fake a green
end-to-end OAuth test.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import anyio
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import (
    OAuthClientInformationFull,
    OAuthClientMetadata,
    OAuthToken,
)

from .diag import log

# How long the interactive login waits for the human to approve in the browser and the
# loopback redirect to land, before giving up. Bounds the flow so an abandoned or
# headless login degrades (no tools + a clear message) instead of hanging the agent.
_LOGIN_TIMEOUT = 300.0
# Poll granularity for the one-shot callback server's blocking accept loop.
_POLL = 1.0


# ---------------------------------------------------------------------------
# Feature flag — ALL OAuth code is gated here. Off by default.
# ---------------------------------------------------------------------------

def oauth_enabled() -> bool:
    """True only when the operator has opted into OAuth via `CHAD_MCP_OAUTH`. With this
    off, no OAuth code runs and the stdio/bearer/HTTP paths behave exactly as plan 016."""
    val = os.environ.get("CHAD_MCP_OAUTH", "")
    return val.strip().lower() not in ("", "0", "false", "no", "off")


def is_oauth(spec: dict) -> bool:
    """Whether a server spec opts into OAuth (an `auth: oauth` marker on an http
    server). Independent of the feature flag — the flag decides whether we act on it."""
    return isinstance(spec, dict) and str(spec.get("auth", "")).lower() == "oauth"


# ---------------------------------------------------------------------------
# Token storage — ~/.chad/mcp_tokens.json, mode 0600, keyed per server
# ---------------------------------------------------------------------------
#
# The file maps a server key -> {"tokens": <OAuthToken>, "client_info":
# <OAuthClientInformationFull>}. Both are pydantic models persisted via model_dump and
# rehydrated via model_validate. The file is created 0600 from the first write so a
# token is never briefly world-readable, mirroring plan 015's trust store. Token VALUES
# are never logged — only the server key and field name ever appear in diagnostics.

_io_lock = threading.Lock()


def tokens_path() -> str:
    # Uses the same os.path.expanduser indirection the rest of mcp.py relies on so tests
    # isolate HOME by monkeypatching expanduser.
    return os.path.join(os.path.expanduser("~"), ".chad", "mcp_tokens.json")


def _read_all() -> dict:
    path = tokens_path()
    try:
        with open(path, "r", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_all(data: dict) -> None:
    path = tokens_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Create with 0600 from the start so credentials are never briefly world-readable.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.chmod(path, 0o600)


def has_tokens(server_key: str) -> bool:
    """Sync helper for the connect-time 'needs login?' decision: does the store hold an
    access token for this server? (Presence only — never inspects/logs the value.)"""
    entry = _read_all().get(server_key)
    return bool(isinstance(entry, dict) and (entry.get("tokens") or {}).get("access_token"))


class FileTokenStorage:
    """`mcp.client.auth.TokenStorage` backed by ~/.chad/mcp_tokens.json (0600).

    The SDK calls these on the portal's event loop, hence async; the file I/O itself is
    plain sync inside (it's small and local). Refreshed tokens flow back through
    `set_tokens` so the SDK's silent refresh is persisted for the next run."""

    def __init__(self, server_key: str):
        self._key = server_key

    async def get_tokens(self) -> "OAuthToken | None":
        entry = _read_all().get(self._key) or {}
        raw = entry.get("tokens")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthToken.model_validate(raw)
        except Exception as e:  # noqa: BLE001 — corrupt entry shouldn't crash connect
            log.warning("mcp: %s stored tokens unreadable (%s)", self._key, type(e).__name__)
            return None

    async def set_tokens(self, tokens: "OAuthToken") -> None:
        with _io_lock:
            data = _read_all()
            entry = data.get(self._key)
            if not isinstance(entry, dict):
                entry = {}
            entry["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
            data[self._key] = entry
            _write_all(data)

    async def get_client_info(self) -> "OAuthClientInformationFull | None":
        entry = _read_all().get(self._key) or {}
        raw = entry.get("client_info")
        if not isinstance(raw, dict):
            return None
        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception as e:  # noqa: BLE001
            log.warning("mcp: %s stored client_info unreadable (%s)", self._key, type(e).__name__)
            return None

    async def set_client_info(self, client_info: "OAuthClientInformationFull") -> None:
        with _io_lock:
            data = _read_all()
            entry = data.get(self._key)
            if not isinstance(entry, dict):
                entry = {}
            entry["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
            data[self._key] = entry
            _write_all(data)


# ---------------------------------------------------------------------------
# Loopback callback server — catches the OAuth redirect on 127.0.0.1
# ---------------------------------------------------------------------------

_CLOSE_PAGE = (
    b"<!doctype html><html><body style='font-family:sans-serif'>"
    b"<h3>chad: login complete</h3><p>You can close this tab.</p></body></html>"
)


class _CallbackHandler(BaseHTTPRequestHandler):
    """One-shot handler: records the redirect's code/state onto the server object and
    shows a 'you can close this tab' page. Silenced so it never spews to the console."""

    def do_GET(self):  # noqa: N802 — http.server API
        qs = parse_qs(urlparse(self.path).query)
        self.server.code = (qs.get("code") or [None])[0]
        self.server.state = (qs.get("state") or [None])[0]
        self.server.error = (qs.get("error") or [None])[0]
        self.server.received = True
        body = _CLOSE_PAGE
        if self.server.error:
            body = (b"<!doctype html><html><body style='font-family:sans-serif'>"
                    b"<h3>chad: login failed</h3><p>You can close this tab.</p></body></html>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # noqa: A002 — silence default request logging
        pass


class LoopbackServer:
    """A localhost HTTP listener bound to an ephemeral port. The bound port is known
    *before* the flow starts, so the redirect URI registered with the provider matches
    exactly what we listen on (no port race). `wait()` blocks for one redirect, bounded
    by a timeout so an abandoned login can't hang forever."""

    def __init__(self):
        self._httpd = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        self._httpd.code = None
        self._httpd.state = None
        self._httpd.error = None
        self._httpd.received = False
        self._httpd.timeout = _POLL
        self.port = self._httpd.server_address[1]
        self.redirect_uri = f"http://127.0.0.1:{self.port}/callback"

    def wait(self, timeout: float):
        """Block until one redirect lands; return (code, state) or None on timeout.
        Sync (run via anyio.to_thread.run_sync off the portal loop)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            # handle_request returns after one request OR after self.timeout seconds
            # (handle_timeout), so this loop stays responsive to the deadline.
            self._httpd.handle_request()
            if self._httpd.received:
                return self._httpd.code, self._httpd.state
        return None

    def close(self):
        try:
            self._httpd.server_close()
        except Exception:  # noqa: BLE001 — best-effort
            pass


# ---------------------------------------------------------------------------
# OAuthClientProvider builders
# ---------------------------------------------------------------------------

def _client_metadata(redirect_uri: str, scope: str | None) -> "OAuthClientMetadata":
    return OAuthClientMetadata(
        # pydantic coerces the str to AnyUrl at construction; the annotation is stricter.
        redirect_uris=[redirect_uri],  # type: ignore[list-item]
        client_name="chad",
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
        scope=scope or None,
    )


def make_noninteractive_provider(server_url: str, storage: "FileTokenStorage",
                                 scope: str | None = None) -> "OAuthClientProvider":
    """Provider for AUTO-CONNECT when tokens already exist: the SDK uses/refreshes the
    stored token silently. The handlers deliberately RAISE — if a refresh fails and the
    SDK would otherwise open a browser, we degrade (connect errors, no tools) instead of
    hanging an agent turn or eval. The operator re-runs `/mcp login` to re-authorize."""

    async def _no_interaction(*_a, **_k):
        raise RuntimeError("interactive OAuth login required — run /mcp login <server>")

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata("http://127.0.0.1/callback", scope),
        storage=storage,
        redirect_handler=_no_interaction,
        callback_handler=_no_interaction,
    )


def make_login_provider(server_url: str, storage: "FileTokenStorage",
                        loopback: "LoopbackServer", scope: str | None = None,
                        emit=None, timeout: float = _LOGIN_TIMEOUT) -> "OAuthClientProvider":
    """Provider for the INTERACTIVE `/mcp login` flow: opens the browser (printing the
    URL as a headless fallback) and catches the redirect on the loopback server. Bounded
    by `timeout` so an abandoned/headless login fails cleanly instead of hanging."""
    say = emit or (lambda _m: None)

    async def redirect_handler(auth_url: str) -> None:
        opened = False
        try:
            import webbrowser
            opened = bool(webbrowser.open(auth_url))
        except Exception:  # noqa: BLE001 — no display / no browser
            opened = False
        if opened:
            say("Opened your browser to approve access. If it didn't open, visit:")
        else:
            say("No browser available. Open this URL to approve access:")
        say(auth_url)

    async def callback_handler() -> "tuple[str, str | None]":
        res = await anyio.to_thread.run_sync(loopback.wait, timeout)
        if res is None:
            raise TimeoutError(f"OAuth login timed out after {int(timeout)}s")
        code, state = res
        if not code:
            raise RuntimeError("OAuth login did not return an authorization code")
        return code, state

    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=_client_metadata(loopback.redirect_uri, scope),
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )
