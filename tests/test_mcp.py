"""Tests for MCP (Model Context Protocol) support (src/chad/mcp.py) + integration.

A real fake MCP server (a tiny stdio JSON-RPC script written to a tmp dir) is spawned
so the whole transport is exercised end to end — the `initialize` handshake, paginated
`tools/list`, `tools/call`, error results, and graceful teardown — not just mocked.

Covers the four client responsibilities (configure, connect, expose, dispatch) plus the
cross-cutting harness wiring: namespaced schema exposure through tools.active_schemas,
dispatch routing through tools.dispatch_for, the read-only/mutating confirm gate, and
the validate.py schema fallback (coercion + unknown-tool listing with no duplication).
"""

import json
import os
import socket
import textwrap
import threading
import time

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from chad import mcp, render, tools, validate

# A minimal but real MCP stdio server: newline-delimited JSON-RPC 2.0. It exposes one
# read-only tool (`echo`, annotated readOnlyHint) and one mutating tool (`write_note`),
# pages tools/list into two pages to exercise the cursor loop, and reports an error for
# a sentinel argument. Pure stdlib so it runs anywhere the test suite does.
_SERVER = textwrap.dedent('''\
    import json, sys

    TOOLS = [
        {"name": "echo",
         "description": "Echo back the message.",
         "inputSchema": {"type": "object",
                         "properties": {"message": {"type": "string"},
                                        "times": {"type": "integer"}},
                         "required": ["message"]},
         "annotations": {"readOnlyHint": True}},
        {"name": "write_note",
         "description": "Persist a note (has a side effect).",
         "inputSchema": {"type": "object",
                         "properties": {"text": {"type": "string"}},
                         "required": ["text"]}},
    ]

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        mid = req.get("id")
        method = req.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": req["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"}}})
        elif method == "notifications/initialized":
            pass  # notification, no reply
        elif method == "tools/list":
            cursor = (req.get("params") or {}).get("cursor")
            if not cursor:
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"tools": TOOLS[:1], "nextCursor": "page2"}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS[1:]}})
        elif method == "tools/call":
            name = req["params"]["name"]
            args = req["params"].get("arguments") or {}
            if args.get("message") == "HANG":
                continue  # never reply: exercise the client-side call timeout
            if args.get("message") == "CRASH":
                sys.exit(0)  # die mid-session (after initialize/list) to test degradation
            if args.get("message") == "BOOM":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "kaboom"}], "isError": True}})
            elif name == "echo":
                txt = str(args.get("message", "")) * int(args.get("times", 1))
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"content": [{"type": "text", "text": txt}]}})
            elif name == "write_note":
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"content": [{"type": "text", "text": "saved"}]}})
            else:
                send({"jsonrpc": "2.0", "id": mid,
                      "error": {"code": -32601, "message": "unknown tool"}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "method not found"}})
''')


# A server that lists the SAME tool name on both pages of tools/list, so the registry's
# duplicate-name guard (mcp._register: `if full in self.by_tool`) fires within one server.
_DUP_SERVER = textwrap.dedent('''\
    import json, sys

    DUP = {"name": "dup", "description": "dup tool",
           "inputSchema": {"type": "object", "properties": {}}}

    def send(msg):
        sys.stdout.write(json.dumps(msg) + "\\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        mid = req.get("id")
        method = req.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": req["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "dup", "version": "1.0"}}})
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            cursor = (req.get("params") or {}).get("cursor")
            if not cursor:
                send({"jsonrpc": "2.0", "id": mid,
                      "result": {"tools": [DUP], "nextCursor": "page2"}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [DUP]}})
        elif method == "tools/call":
            send({"jsonrpc": "2.0", "id": mid,
                  "result": {"content": [{"type": "text", "text": "ok"}]}})
        else:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "method not found"}})
''')


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project dir with a .mcp.json pointing at the fake server, an isolated HOME so
    the developer's real ~/.chad/mcp.json never leaks in, and cwd set to the project.
    Tears down MCP connections before and after so no registry/process leaks across tests."""
    server_py = tmp_path / "server.py"
    server_py.write_text(_SERVER)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    config = {"mcpServers": {"demo": {"command": "python", "args": [str(server_py)]}}}
    (tmp_path / ".mcp.json").write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    # The demo server is project-scoped (./.mcp.json), which is gated behind the trust
    # store. Trust this project so the connect/dispatch tests exercise a live server;
    # the untrusted-path behavior is covered by its own tests below.
    mcp._set_trusted(str(tmp_path))
    mcp.reset_session()
    yield tmp_path
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Connect + expose
# ---------------------------------------------------------------------------

def test_connect_lists_paginated_tools(project):
    reg = mcp.service()
    assert reg.tool_names() == ["mcp__demo__echo", "mcp__demo__write_note"]
    # both pages of tools/list were collected
    assert len(reg.clients) == 1 and not reg.clients[0].error


def test_schemas_namespaced_and_param_preserved(project):
    schemas = mcp.schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == ["mcp__demo__echo", "mcp__demo__write_note"]
    echo = next(s for s in schemas if s["function"]["name"] == "mcp__demo__echo")
    assert echo["function"]["parameters"]["required"] == ["message"]
    assert "MCP server 'demo'" in echo["function"]["description"]


def test_active_schemas_includes_mcp(project):
    names = [s["function"]["name"] for s in tools.active_schemas()]
    assert "mcp__demo__echo" in names
    assert "bash" in names  # builtins still present


def test_no_config_no_tools(tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    monkeypatch.chdir(tmp_path)
    mcp.reset_session()
    assert mcp.schemas() == []
    assert "no MCP servers configured" in mcp.summary_lines()[0]
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def test_dispatch_round_trip(project):
    fn = tools.dispatch_for("mcp__demo__echo")
    assert fn is not None
    assert fn({"message": "hi", "times": 3}) == "hihihi"


def test_dispatch_error_result_prefixed(project):
    out = tools.dispatch_for("mcp__demo__echo")({"message": "BOOM"})
    assert out.startswith("[tool reported an error]")
    assert "kaboom" in out


def test_unknown_mcp_name_not_dispatched(project):
    assert tools.dispatch_for("mcp__demo__nope") is None
    assert tools.dispatch_for("mcp__ghost__x") is None


# ---------------------------------------------------------------------------
# Mutating / confirm classification
# ---------------------------------------------------------------------------

def test_readonly_hint_skips_confirm(project):
    assert tools.is_mutating("mcp__demo__echo") is False        # readOnlyHint: true
    assert tools.is_mutating("mcp__demo__write_note") is True   # default: mutating
    assert tools.is_mutating("bash") is True                    # builtin unaffected


# ---------------------------------------------------------------------------
# validate.py integration (schema fallback, no duplication)
# ---------------------------------------------------------------------------

def test_validate_coerces_mcp_args(project):
    # "3" -> 3 against the MCP tool's integer param, exactly like a builtin.
    coerced, errs = validate.coerce_and_validate("mcp__demo__echo",
                                                 {"message": "x", "times": "3"})
    assert errs == [] and coerced["times"] == 3


def test_validate_flags_missing_required_mcp_arg(project):
    _, errs = validate.coerce_and_validate("mcp__demo__echo", {"times": 1})
    assert any(e.path.endswith("message") and e.got == "missing" for e in errs)


def test_unknown_tool_listing_includes_mcp(project):
    msg = validate.render_repair("totally_bogus", {}, [])
    assert "mcp__demo__echo" in msg
    assert "bash" in msg  # builtins listed too


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_confirm_preview_shows_mcp_args():
    preview = render.confirm_preview("mcp__demo__write_note", {"text": "hello"})
    assert "hello" in preview


# ---------------------------------------------------------------------------
# Config merge + robustness
# ---------------------------------------------------------------------------

def test_project_overrides_user(tmp_path, monkeypatch):
    home = tmp_path / "home"; (home / ".chad").mkdir(parents=True)
    (home / ".chad" / "mcp.json").write_text(json.dumps(
        {"mcpServers": {"demo": {"command": "user-cmd"},
                        "useronly": {"command": "u"}}}))
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    (tmp_path / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"demo": {"command": "project-cmd"}}}))
    servers, _ = mcp._load_config(str(tmp_path), str(home))
    by = {n: s for n, s, _ in servers}
    scope = {n: sc for n, s, sc in servers}
    assert by["demo"]["command"] == "project-cmd"   # project wins
    assert "useronly" in by                         # user-only kept
    assert scope["demo"] == "project"               # merged entry takes project scope
    assert scope["useronly"] == "user"              # user-only keeps user scope


def test_malformed_config_is_skipped(tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    (tmp_path / ".mcp.json").write_text("{ this is not json ")
    monkeypatch.chdir(tmp_path)
    mcp.reset_session()
    assert mcp.schemas() == []          # no crash; just no tools
    assert mcp.service().warnings       # warning recorded
    mcp.reset_session()


def test_disabled_server_skipped(tmp_path, monkeypatch):
    server_py = tmp_path / "server.py"; server_py.write_text(_SERVER)
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    (tmp_path / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"demo": {"command": "python", "args": [str(server_py)],
                                 "disabled": True}}}))
    monkeypatch.chdir(tmp_path)
    mcp.reset_session()
    assert mcp.schemas() == []
    mcp.reset_session()


def test_bad_command_degrades_gracefully(tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir()
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    (tmp_path / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"broken": {"command": "this_binary_does_not_exist_chad"}}}))
    monkeypatch.chdir(tmp_path)
    mcp.reset_session()
    assert mcp.schemas() == []                       # no tools from a dead server
    assert any("broken" in w for w in mcp.service().warnings)
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Trust gate (project-level ./.mcp.json is untrusted until /mcp trust)
# ---------------------------------------------------------------------------

def _isolate_home(tmp_path, monkeypatch):
    """Point ~ at an isolated dir so the developer's real ~/.chad never leaks in.
    Returns the home Path."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    return home


def _write_project(tmp_path, server_py, name="demo", server_src=_SERVER, **spec):
    server_py.write_text(server_src)
    full_spec = {"command": "python", "args": [str(server_py)]}
    full_spec.update(spec)
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {name: full_spec}}))


def test_untrusted_project_server_not_started(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    _write_project(tmp_path, tmp_path / "server.py")
    monkeypatch.chdir(tmp_path)
    mcp.reset_session()
    # A project-only server in an untrusted dir must NOT launch.
    assert mcp.schemas() == []
    reg = mcp.service()
    assert [n for n, _ in reg.blocked] == ["demo"]
    assert not reg.clients                            # the subprocess was never spawned
    assert any("not trusted" in w for w in reg.warnings)
    assert any("blocked" in ln for ln in mcp.summary_lines())
    mcp.reset_session()


def test_trust_enables_project_server(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    _write_project(tmp_path, tmp_path / "server.py")
    monkeypatch.chdir(tmp_path)
    mcp.reset_session()
    assert mcp.schemas() == []                        # untrusted: nothing exposed
    mcp.trust()                                        # opt in + reset
    names = [s["function"]["name"] for s in mcp.schemas()]
    assert "mcp__demo__echo" in names                 # now the server's tools appear
    assert not mcp.service().blocked
    mcp.reset_session()


def test_user_level_server_trusted_by_default(tmp_path, monkeypatch):
    home = _isolate_home(tmp_path, monkeypatch)
    (home / ".chad").mkdir(parents=True)
    server_py = tmp_path / "server.py"; server_py.write_text(_SERVER)
    (home / ".chad" / "mcp.json").write_text(json.dumps(
        {"mcpServers": {"udemo": {"command": "python", "args": [str(server_py)]}}}))
    workdir = tmp_path / "work"; workdir.mkdir()
    monkeypatch.chdir(workdir)                         # no project ./.mcp.json here
    mcp.reset_session()
    names = [s["function"]["name"] for s in mcp.schemas()]
    assert "mcp__udemo__echo" in names                 # user-level connects, no trust step
    assert not mcp.service().blocked
    mcp.reset_session()


# ---------------------------------------------------------------------------
# Registry robustness at the seam
# ---------------------------------------------------------------------------

def test_duplicate_tool_name_across_servers_warns(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    _write_project(tmp_path, tmp_path / "dup.py", name="ds", server_src=_DUP_SERVER)
    monkeypatch.chdir(tmp_path)
    mcp._set_trusted(str(tmp_path))
    mcp.reset_session()
    reg = mcp.service()
    assert reg.tool_names() == ["mcp__ds__dup"]        # the dup copy is dropped, first wins
    assert any("duplicate tool name" in w for w in reg.warnings)
    mcp.reset_session()


def test_server_name_with_separator_rejected(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    _write_project(tmp_path, tmp_path / "server.py", name="a__b")
    monkeypatch.chdir(tmp_path)
    mcp._set_trusted(str(tmp_path))
    mcp.reset_session()
    assert mcp.schemas() == []                         # the '__'-named server is skipped
    assert any("a__b" in w and "__" in w for w in mcp.service().warnings)
    mcp.reset_session()


def test_result_truncated_to_cap(project):
    big = mcp._RESULT_MAX_CHARS + 5000
    out = tools.dispatch_for("mcp__demo__echo")({"message": "x", "times": big})
    assert len(out) == mcp._RESULT_MAX_CHARS           # clamped to the cap, not 25000


def test_call_timeout_surfaces_error(tmp_path, monkeypatch):
    _isolate_home(tmp_path, monkeypatch)
    # A tiny per-server timeout so the unanswered call returns fast, no test hang.
    _write_project(tmp_path, tmp_path / "server.py", timeout=0.5)
    monkeypatch.chdir(tmp_path)
    mcp._set_trusted(str(tmp_path))
    mcp.reset_session()
    out = tools.dispatch_for("mcp__demo__echo")({"message": "HANG"})
    assert "MCP error" in out and "TimeoutError" in out
    mcp.reset_session()


def test_midsession_crash_degrades(project):
    # Server exits during the call; a transport error comes back as text, never raises.
    out = tools.dispatch_for("mcp__demo__echo")({"message": "CRASH"})
    assert out.startswith("[MCP error")
    # A follow-up call against the now-dead server also degrades gracefully.
    out2 = tools.dispatch_for("mcp__demo__echo")({"message": "again"})
    assert out2.startswith("[MCP error")


# ---------------------------------------------------------------------------
# HTTP transport (Streamable HTTP) — a real in-process MCP server on loopback
# ---------------------------------------------------------------------------
#
# The SDK's `streamablehttp_client` drives a true end-to-end HTTP transport against a
# FastMCP server (built with the SDK's server side) run by uvicorn in a background
# thread on an ephemeral port. This exercises the same public seam as the stdio tests,
# proving the transport swap is behavior-preserving across both transports.

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _CaptureAuth:
    """Pure-ASGI middleware that records the Authorization header of each request, so a
    test can prove the bearer token from config actually reached the server."""

    def __init__(self, app, holder):
        self.app, self.holder = app, holder

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            for k, v in scope.get("headers", []):
                if k == b"authorization":
                    self.holder["auth"] = v.decode()
        await self.app(scope, receive, send)


def _build_http_app(holder):
    """A Streamable HTTP MCP app exposing one read-only tool (`ping`, readOnlyHint),
    one mutating tool (`do_write`), and one that always errors (`boom`)."""
    server = FastMCP("httpdemo")

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def ping(message: str) -> str:
        """Echo back the message (read-only)."""
        return "pong:" + message

    @server.tool()
    def do_write(text: str) -> str:
        """Persist a note (has a side effect)."""
        return "wrote:" + text

    @server.tool()
    def boom() -> str:
        """Always fails, to exercise the isError path."""
        raise ValueError("kaboom")

    return _CaptureAuth(server.streamable_http_app(), holder)


class _HttpServer:
    """A real Streamable HTTP MCP server on a loopback port, run by uvicorn in a daemon
    thread. `holder["auth"]` captures the last Authorization header the server saw."""

    def __init__(self):
        self.holder = {"auth": None}
        self.port = _free_port()
        cfg = uvicorn.Config(_build_http_app(self.holder), host="127.0.0.1",
                             port=self.port, log_level="warning")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/mcp"

    def start(self):
        self._thread.start()
        for _ in range(250):          # up to ~5s for the loop to come up
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


def _http_config(tmp_path, monkeypatch, url, **extra):
    """Isolate HOME, write a trusted project ./.mcp.json with one HTTP server, chdir, and
    reset the session so the next service() connects over HTTP. Returns nothing."""
    home = tmp_path / "home"; home.mkdir(exist_ok=True)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    spec = {"type": "http", "url": url}
    spec.update(extra)
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"web": spec}}))
    monkeypatch.chdir(tmp_path)
    mcp._set_trusted(str(tmp_path))
    mcp.reset_session()


def test_http_connect_and_list(http_server, tmp_path, monkeypatch):
    _http_config(tmp_path, monkeypatch, http_server.url,
                 headers={"Authorization": "Bearer test-token-123"})
    reg = mcp.service()
    names = reg.tool_names()
    assert "mcp__web__ping" in names and "mcp__web__do_write" in names
    assert reg.clients[0].transport == "http"           # HTTP transport chosen by `url`
    assert reg.is_mutating("mcp__web__ping") is False    # readOnlyHint -> auto-run
    assert reg.is_mutating("mcp__web__do_write") is True  # default -> mutating
    mcp.reset_session()


def test_http_call_round_trip(http_server, tmp_path, monkeypatch):
    _http_config(tmp_path, monkeypatch, http_server.url,
                 headers={"Authorization": "Bearer test-token-123"})
    out = tools.dispatch_for("mcp__web__ping")({"message": "hi"})
    assert out == "pong:hi"
    mcp.reset_session()


def test_http_error_result_prefixed(http_server, tmp_path, monkeypatch):
    _http_config(tmp_path, monkeypatch, http_server.url,
                 headers={"Authorization": "Bearer test-token-123"})
    out = tools.dispatch_for("mcp__web__boom")({})
    assert out.startswith("[tool reported an error]")    # isError honored over HTTP
    mcp.reset_session()


def test_http_bearer_header_sent(http_server, tmp_path, monkeypatch):
    _http_config(tmp_path, monkeypatch, http_server.url,
                 headers={"Authorization": "Bearer test-token-123"})
    mcp.service()  # connecting drives requests that must carry the configured header
    assert http_server.holder["auth"] == "Bearer test-token-123"
    mcp.reset_session()


def test_http_connect_timeout_degrades(tmp_path, monkeypatch):
    dead = _free_port()                                   # nothing listening here
    start = time.time()
    _http_config(tmp_path, monkeypatch, f"http://127.0.0.1:{dead}/mcp",
                 connect_timeout=3)
    reg = mcp.service()
    assert reg.tool_names() == []                         # no tools from a dead endpoint
    assert any("web" in w for w in reg.warnings)          # warning recorded
    assert time.time() - start < 15                       # bounded: didn't hang
    mcp.reset_session()


def test_dead_endpoint_does_not_block_healthy_server(http_server, tmp_path, monkeypatch):
    home = tmp_path / "home"; home.mkdir(exist_ok=True)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(home) if p == "~" or p.startswith("~/") else p)
    dead = _free_port()
    cfg = {"mcpServers": {
        "web": {"type": "http", "url": http_server.url},
        "dead": {"type": "http", "url": f"http://127.0.0.1:{dead}/mcp",
                 "connect_timeout": 3},
    }}
    (tmp_path / ".mcp.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)
    mcp._set_trusted(str(tmp_path))
    mcp.reset_session()
    reg = mcp.service()
    names = reg.tool_names()
    assert "mcp__web__ping" in names                      # healthy server unaffected
    assert all(not n.startswith("mcp__dead__") for n in names)  # dead one contributed none
    assert any("dead" in w for w in reg.warnings)         # parallel + isolated connect
    mcp.reset_session()


# ---------------------------------------------------------------------------
# stdio subprocess environment filter (plan 044 item 6)
# ---------------------------------------------------------------------------

def test_stdio_env_withholds_secrets_by_default():
    # A parent environment carrying a secret + the essentials the subprocess needs.
    parent = {"PATH": "/usr/bin", "HOME": "/home/me", "LANG": "en_US.UTF-8",
              "TERM": "xterm", "OPENAI_API_KEY": "sk-secret", "AWS_SECRET_KEY": "top"}
    env = mcp._stdio_env(parent, None)
    assert env["PATH"] == "/usr/bin"                 # essentials pass through
    assert env["HOME"] == "/home/me"
    assert env["LANG"] == "en_US.UTF-8"
    assert "OPENAI_API_KEY" not in env               # secrets withheld
    assert "AWS_SECRET_KEY" not in env


def test_stdio_env_config_extra_merges_and_wins():
    # A server's own `env:` block is always honored and overrides the inherited value.
    parent = {"PATH": "/usr/bin", "HOME": "/home/me"}
    env = mcp._stdio_env(parent, {"MY_TOKEN": "abc", "PATH": "/custom/bin"})
    assert env["MY_TOKEN"] == "abc"                  # explicitly declared var passes
    assert env["PATH"] == "/custom/bin"              # config `env:` overrides inherited
    assert env["HOME"] == "/home/me"


def test_stdio_env_full_env_escape_hatch():
    # CHAD_MCP_FULL_ENV=1 restores the historical full inherit for the rare server.
    parent = {"PATH": "/usr/bin", "SECRET": "s", "CHAD_MCP_FULL_ENV": "1"}
    env = mcp._stdio_env(parent, None)
    assert env["SECRET"] == "s"                      # everything inherited
    assert env["PATH"] == "/usr/bin"


def test_stdio_env_extra_non_dict_ignored():
    # A malformed `env:` (not an object) must not crash the transport build.
    env = mcp._stdio_env({"PATH": "/usr/bin"}, "not-a-dict")
    assert env == {"PATH": "/usr/bin"}
