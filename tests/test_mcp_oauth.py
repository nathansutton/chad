"""Tests for OAuth support on hosted HTTP MCP servers (src/chad/mcp_oauth.py + the
plan-017 wiring in src/chad/mcp.py).

What is tested here are the SEAMS chad owns, not the OAuth protocol itself:
  - the TokenStorage round-trip through ~/.chad/mcp_tokens.json, its 0600 mode, and the
    missing-token path (Step 2);
  - the loopback callback server parsing a fired ?code=&state= redirect and stopping
    (Step 3);
  - the feature flag: with CHAD_MCP_OAUTH off, an `auth: oauth` server contributes no
    tools and surfaces a warning, and nothing else changes vs plan 016 (Step 1);
  - the "needs login" state when the flag is on but no tokens are stored, and that the
    server is NOT auto-connected (Step 4);
  - that once tokens exist the server connects non-interactively and its tools appear,
    driven against a real in-process HTTP MCP server that does not itself require auth
    (so no real IdP is needed — the OAuth provider just attaches the stored token);
  - headless / no-browser graceful degradation: a login with the browser unavailable and
    the callback timing out fast returns a clear failure and never raises or hangs (Step 5).

E2E GAP (intentional): a full 3-legged OAuth handshake against a real authorization
server is NOT exercised — there is no real IdP in CI. We never fake a green end-to-end
OAuth test. No real token value appears anywhere in this file; all tokens are obvious
fakes (the literal string "fake").
"""

import json
import os
import socket
import stat
import threading
import time
import urllib.request

import anyio
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from mcp.types import ToolAnnotations

from chad import mcp, mcp_oauth

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isolate_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    return home


def _enable_flag(monkeypatch):
    monkeypatch.setenv("CHAD_MCP_OAUTH", "1")


def _fake_token():
    # Obvious fake; never a real credential. exclude_none keeps the stored shape lean.
    return OAuthToken(access_token="fake", token_type="Bearer", refresh_token="fake")


def _store_fake_tokens(server_key):
    """Seed the token store with fake tokens for a server (drives the tokens-present
    path without a real auth flow)."""
    storage = mcp_oauth.FileTokenStorage(server_key)
    anyio.run(storage.set_tokens, _fake_token())


# ---------------------------------------------------------------------------
# Step 2: TokenStorage round-trip, 0600 perms, missing-token path
# ---------------------------------------------------------------------------

def test_token_storage_round_trip(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    storage = mcp_oauth.FileTokenStorage("svc")

    async def go():
        assert await storage.get_tokens() is None          # missing-token path
        assert await storage.get_client_info() is None
        await storage.set_tokens(_fake_token())
        await storage.set_client_info(OAuthClientInformationFull(
            client_id="cid-fake", redirect_uris=["http://127.0.0.1:9/callback"]))
        tok = await storage.get_tokens()
        ci = await storage.get_client_info()
        assert tok is not None and tok.access_token == "fake"
        assert ci is not None and ci.client_id == "cid-fake"

    anyio.run(go)


def test_token_file_is_0600(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    storage = mcp_oauth.FileTokenStorage("svc")
    anyio.run(storage.set_tokens, _fake_token())
    path = mcp_oauth.tokens_path()
    assert os.path.isfile(path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_missing_token_returns_none_and_has_tokens_false(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    assert mcp_oauth.has_tokens("never-logged-in") is False
    assert anyio.run(mcp_oauth.FileTokenStorage("never-logged-in").get_tokens) is None
    _store_fake_tokens("svc")
    assert mcp_oauth.has_tokens("svc") is True
    assert mcp_oauth.has_tokens("other") is False          # per-server isolation


def test_no_token_literal_leaks_in_diag(tmp_path, monkeypatch, caplog):
    """A corrupt entry warns by server key + field only — never the value."""
    _isolate_home(tmp_path, monkeypatch)
    path = mcp_oauth.tokens_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"svc": {"tokens": {"not": "valid-token-shape"}}}, f)
    with caplog.at_level("WARNING"):
        assert anyio.run(mcp_oauth.FileTokenStorage("svc").get_tokens) is None
    assert "valid-token-shape" not in caplog.text


# ---------------------------------------------------------------------------
# Step 3: loopback callback server parses a fired redirect and stops
# ---------------------------------------------------------------------------

def test_callback_server_parses_redirect_and_stops():
    lb = mcp_oauth.LoopbackServer()
    assert lb.redirect_uri.startswith("http://127.0.0.1:")
    result = {}

    def waiter():
        result["res"] = lb.wait(timeout=10)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.2)
    # Fire the redirect the IdP would send back to the loopback URI.
    url = lb.redirect_uri + "?code=the-auth-code&state=xyz"
    body = urllib.request.urlopen(url, timeout=5).read()
    assert b"close this tab" in body
    t.join(timeout=10)
    lb.close()
    assert result["res"] == ("the-auth-code", "xyz")


def test_callback_server_times_out_without_request():
    lb = mcp_oauth.LoopbackServer()
    t0 = time.monotonic()
    res = lb.wait(timeout=1.0)
    lb.close()
    assert res is None
    assert time.monotonic() - t0 < 5          # bounded, did not hang


# ---------------------------------------------------------------------------
# Step 1: feature flag OFF — oauth server skipped with a warning, nothing else changes
# ---------------------------------------------------------------------------

def _write_user_oauth_server(home, name="hosted", url="http://127.0.0.1:9/mcp"):
    (home / ".chad").mkdir(parents=True, exist_ok=True)
    (home / ".chad" / "mcp.json").write_text(json.dumps(
        {"mcpServers": {name: {"type": "http", "url": url, "auth": "oauth"}}}))


def test_flag_off_oauth_server_skipped(tmp_path, monkeypatch):
    home = _isolate_home(tmp_path, monkeypatch)
    _write_user_oauth_server(home)
    workdir = tmp_path / "work"; workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("CHAD_MCP_OAUTH", raising=False)
    mcp.reset_session()
    reg = mcp.service()
    assert mcp.schemas() == []                          # no tools
    assert not reg.clients                              # never attempted a connect
    assert reg.needs_login == []
    assert any("OAuth disabled" in w for w in reg.warnings)
    assert any("OAuth disabled" in ln for ln in mcp.summary_lines())
    mcp.reset_session()


def test_flag_off_does_not_disturb_plain_http(tmp_path, monkeypatch):
    """A non-oauth server alongside an oauth one is unaffected by the flag-off skip."""
    home = _isolate_home(tmp_path, monkeypatch)
    (home / ".chad").mkdir(parents=True, exist_ok=True)
    (home / ".chad" / "mcp.json").write_text(json.dumps({"mcpServers": {
        "oauthsrv": {"type": "http", "url": "http://127.0.0.1:9/mcp", "auth": "oauth"},
        # a stdio server that fails to launch still proves the non-oauth path is intact
        "plain": {"command": "this_binary_does_not_exist_chad"},
    }}))
    workdir = tmp_path / "work"; workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("CHAD_MCP_OAUTH", raising=False)
    mcp.reset_session()
    reg = mcp.service()
    # plain server was attempted (a client recorded, errored); oauth server was skipped.
    assert [c.name for c in reg.clients] == ["plain"]
    assert any("OAuth disabled" in w for w in reg.warnings)
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Step 4: needs-login state (flag on, no tokens) — does NOT auto-connect
# ---------------------------------------------------------------------------

def test_needs_login_when_no_tokens(tmp_path, monkeypatch):
    home = _isolate_home(tmp_path, monkeypatch)
    _write_user_oauth_server(home)
    workdir = tmp_path / "work"; workdir.mkdir()
    monkeypatch.chdir(workdir)
    _enable_flag(monkeypatch)
    mcp.reset_session()
    reg = mcp.service()
    assert mcp.schemas() == []                          # no tools yet
    assert not reg.clients                              # NOT auto-connected (no browser)
    assert [n for n, _ in reg.needs_login] == ["hosted"]
    assert any("needs login" in w for w in reg.warnings)
    assert any("needs login" in ln for ln in mcp.summary_lines())
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Step 4: tokens present => connects non-interactively, tools appear.
# Driven against a real in-process HTTP MCP server that itself requires no auth, so the
# OAuth provider simply attaches the stored (fake) token and the handshake succeeds —
# no real authorization server is involved.
# ---------------------------------------------------------------------------

def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def _build_app():
    server = FastMCP("oauthdemo")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def ping(message: str) -> str:
        """Echo back the message (read-only)."""
        return "pong:" + message

    return server.streamable_http_app()


class _HttpServer:
    def __init__(self):
        self.port = _free_port()
        cfg = uvicorn.Config(_build_app(), host="127.0.0.1", port=self.port,
                             log_level="warning")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self):
        self._thread.start()
        for _ in range(250):
            if self._server.started:
                return
            time.sleep(0.02)
        raise RuntimeError("HTTP MCP server did not start")

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def http_server():
    srv = _HttpServer()
    srv.start()
    yield srv
    srv.stop()


def test_tokens_present_connects_and_exposes_tools(http_server, tmp_path, monkeypatch):
    home = _isolate_home(tmp_path, monkeypatch)
    _write_user_oauth_server(home, name="hosted", url=http_server.url)
    workdir = tmp_path / "work"; workdir.mkdir()
    monkeypatch.chdir(workdir)
    _enable_flag(monkeypatch)
    _store_fake_tokens("hosted")          # pretend a prior /mcp login succeeded
    mcp.reset_session()
    reg = mcp.service()
    names = [s["function"]["name"] for s in mcp.schemas()]
    assert "mcp__hosted__ping" in names                 # tools appear, no interaction
    assert reg.needs_login == []                        # not gated once tokens exist
    assert reg.clients and reg.clients[0].transport == "http"
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Step 5: headless / no-browser graceful degradation
# ---------------------------------------------------------------------------

def test_login_headless_degrades_gracefully(tmp_path, monkeypatch):
    """No browser available AND the callback never lands: login returns a clear failure
    string, never raises, never hangs. Uses a closed loopback port so the connect fails
    fast; the redirect handler's webbrowser.open is forced to raise."""
    import webbrowser
    home = _isolate_home(tmp_path, monkeypatch)
    # Point at a dead port so the OAuth connect fails quickly rather than waiting on a
    # real authorization server (we are testing degradation, not the happy path).
    _write_user_oauth_server(home, name="hosted", url="http://127.0.0.1:1/mcp")
    workdir = tmp_path / "work"; workdir.mkdir()
    monkeypatch.chdir(workdir)
    _enable_flag(monkeypatch)
    monkeypatch.setattr(webbrowser, "open",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no display")))
    # Shrink the login timeout so an abandoned flow doesn't slow the suite.
    monkeypatch.setattr(mcp_oauth, "_LOGIN_TIMEOUT", 1.0)
    mcp.reset_session()
    msgs = []
    result = mcp.login("hosted", emit=msgs.append)
    assert isinstance(result, str)
    assert "login failed" in result or "did not produce a token" in result
    assert mcp_oauth.has_tokens("hosted") is False       # nothing persisted
    mcp.reset_session()


def test_login_flag_off_message(tmp_path, monkeypatch):
    home = _isolate_home(tmp_path, monkeypatch)
    _write_user_oauth_server(home)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CHAD_MCP_OAUTH", raising=False)
    mcp.reset_session()
    msg = mcp.login("hosted")
    assert "disabled" in msg.lower()
    mcp.reset_session()


def test_login_unknown_server_message(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    _enable_flag(monkeypatch)
    mcp.reset_session()
    assert "no MCP server" in mcp.login("nope")
    mcp.reset_session()
