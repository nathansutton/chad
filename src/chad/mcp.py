"""Model Context Protocol (MCP) client — external tool servers as chad tools.

An *MCP server* is a separate process (or hosted HTTP endpoint) that exposes tools
(and resources/prompts we don't use yet). Connecting one lets chad call tools it
doesn't ship: a GitHub server, a Postgres server, Linear/Slack/Atlassian, a
company's internal API, etc. This is the same extension mechanism Claude Code uses.

Transport is provided by the official `mcp` Python SDK (`stdio_client` for local
subprocess servers, `streamablehttp_client` for hosted HTTP servers). The SDK is
**async (anyio)**; chad's tool layer is **synchronous and blocking**. The bridge is
a single `anyio.from_thread.BlockingPortal` running one event loop in a background
thread, owned by the registry for its lifetime. Each server's transport +
`ClientSession` scopes are kept open on that loop (a connect task that awaits a stop
event), and every `tools/call` is marshalled onto the loop via `portal.call`. The
per-call timeout lives *inside* the coroutine (`anyio.fail_after`) so a hung remote
server cancels the task and frees the agent thread — it can never wedge the agent.

Four client responsibilities, mirroring `skills.py`:

  1. Configure — read `.mcp.json` (project) and `~/.chad/mcp.json` (user); project
                 servers override user servers of the same name.
  2. Connect   — for each server, open the transport, run `initialize`, list tools.
                 Connects run in parallel and time-bounded so one dead endpoint can't
                 stall the others or the first turn.
  3. Expose    — `schemas()` returns OpenAI/Qwen tool definitions named
                 `mcp__<server>__<tool>` (namespaced so they can't collide with
                 chad's builtins or each other), appended to `tools.active_schemas()`.
  4. Dispatch  — `call(name, args)` routes a model tool-call to the owning server and
                 returns the result text.

Everything degrades gracefully: a server that's missing, misconfigured, slow to
connect, or that crashes mid-session contributes no tools and never takes the agent
down — exactly like a language server that won't start in `lsp.py`.

State (the cwd-keyed registry of live connections) lives at module level like
`skills._registry`, is rebuilt when cwd changes, and is torn down by
`reset_session()` on a new Agent / `/reset` and by an atexit hook on process exit so
server subprocesses and HTTP sessions never leak.
"""

import atexit
import json
import os
import threading
from functools import partial
from typing import Any, Optional

import anyio
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# `streamablehttp_client(url, headers=..., auth=...)` is the documented HTTP entry
# point that takes headers (and, in plan 017, an `auth=` OAuth provider) directly,
# building the httpx client via MCP's own factory with the right defaults. mcp 1.28.1
# marks it deprecated in favour of `streamable_http_client(url, *, http_client=...)`,
# but that variant has no `headers=`/`auth=` — you'd hand-build an httpx client and
# risk dropping MCP's required client config — so we keep this one until the new API
# grows an equivalent. (Confirmed against mcp==1.28.1.)
from mcp.client.streamable_http import streamablehttp_client

from . import mcp_oauth
from .diag import log, warn_footer

# Namespacing: a model-visible MCP tool is `mcp__<server>__<tool>`. The separator is
# the Claude-Code convention; it keeps server tools from colliding with chad builtins
# (bash/read/…) or with each other, and lets dispatch recover (server, tool) by split.
PREFIX = "mcp__"
_SEP = "__"

# Timeouts (seconds). Connect (initialize + tools/list) is generous because a stdio
# server launched via `npx`/`uvx` may download the package on first run, and an HTTP
# endpoint may be slow; per-call default is the same 120s as `tool_bash`, overridable
# per server with a "timeout" config key.
_CONNECT_TIMEOUT = 60
_CALL_TIMEOUT = 120
# Extra slack on the main-thread join over the in-coroutine connect timeout, so the
# coroutine's own fail_after fires first and reports a clean per-server error.
_CONNECT_JOIN_GRACE = 5
# A tools/list that pages forever (buggy server) shouldn't hang the session.
_MAX_LIST_PAGES = 50
# Cap a tool result so a chatty server can't blow up the context / prefill budget,
# same spirit as tool_bash's 20k clamp.
_RESULT_MAX_CHARS = 20000


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _config_paths(cwd: str, home: str):
    """Config files to read, lowest precedence first (project overrides user), each
    tagged with its trust scope: user-level is authored by the operator (trusted);
    project-level (`./.mcp.json`) ships with the repo and is attacker-influenced."""
    return [
        (os.path.join(home, ".chad", "mcp.json"), "user"),       # user-level (TRUSTED)
        (os.path.join(cwd, ".mcp.json"), "project"),             # project-level (UNTRUSTED)
    ]


def _load_config(cwd: str = None, home: str = None):
    """Merge MCP server definitions from the config files. Returns an ordered list of
    (name, spec, scope) where scope is "user" or "project", plus a list of
    human-readable warnings. Project entries override user entries of the same name,
    and the merged entry's scope becomes "project" (it now carries attacker-influenced
    fields, so it must clear the trust gate). Lenient: a malformed file warns and is
    skipped, it never aborts startup."""
    cwd = cwd or os.getcwd()
    home = home or os.path.expanduser("~")
    merged = {}
    scopes = {}
    order = []
    warnings = []
    for path, scope in _config_paths(cwd, home):
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", errors="replace") as f:
                data = json.load(f)
        except (OSError, ValueError) as e:
            warnings.append(f"{path}: skipped ({e})")
            log.warning("mcp: cannot read config %s — %s", path, e)
            continue
        servers = data.get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            warnings.append(f"{path}: no 'mcpServers' object")
            continue
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                warnings.append(f"{path}: server {name!r} is not an object")
                continue
            if name not in merged:
                order.append(name)
            merged[name] = spec
            scopes[name] = scope
    return [(n, merged[n], scopes[n]) for n in order], warnings


# ---------------------------------------------------------------------------
# Trust store — gate untrusted project-level servers behind an explicit opt-in
# ---------------------------------------------------------------------------
#
# A `./.mcp.json` ships with a repo and names a subprocess (command/args/env/cwd) or
# an HTTP endpoint we'd otherwise reach on the first agent turn — cloning a hostile
# repo and running chad in it would launch attacker-controlled processes or send
# tokens to an attacker URL with no confirmation. So a project-scope server is only
# started once the operator has trusted that project path (via `/mcp trust`).
# User-level config is authored by the operator and is never gated. Trust is keyed by
# absolute cwd and persisted to ~/.chad/trusted_mcp.json (mode 0600); if a project
# moves dirs it re-prompts, which is intentional (the path is the trust anchor).

def _trust_store_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".chad", "trusted_mcp.json")


def _load_trust() -> dict:
    path = _trust_store_path()
    try:
        with open(path, "r", errors="replace") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _is_trusted(cwd: str) -> bool:
    return _load_trust().get(os.path.abspath(cwd)) is True


def _set_trusted(cwd: str):
    """Mark a project path as trusted, persisting to ~/.chad/trusted_mcp.json (0600)."""
    path = _trust_store_path()
    data = _load_trust()
    data[os.path.abspath(cwd)] = True
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create with 0600 from the start so the trust list isn't briefly world-readable.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.chmod(path, 0o600)
    except OSError as e:
        log.warning("mcp: could not persist trust store %s — %s", path, e)


# ---------------------------------------------------------------------------
# Transport build (stdio | streamable HTTP) — the only auth here is static bearer
# headers; OAuth (auth= provider) slots into the same call sites in plan 017.
# ---------------------------------------------------------------------------

def _transport_for(spec: dict, auth=None):
    """Return (kind, build) for a server spec, where `build()` is a zero-arg factory
    that yields the SDK transport async context manager. Transport is chosen by
    presence: `url` -> HTTP, else `command` -> stdio (`type` is advisory only).
    `auth` is an optional OAuthClientProvider (plan 017) passed straight through to
    `streamablehttp_client(..., auth=...)`; None keeps the plan-016 bearer/header path.
    Returns (None, None) if neither is present."""
    url = spec.get("url")
    if isinstance(url, str) and url:
        headers = spec.get("headers")
        headers = headers if isinstance(headers, dict) else None
        return "http", lambda: streamablehttp_client(url, headers=headers, auth=auth)
    command = spec.get("command")
    if isinstance(command, str) and command:
        env = dict(os.environ)
        extra = spec.get("env")
        if isinstance(extra, dict):
            env.update({k: str(v) for k, v in extra.items()})
        params = StdioServerParameters(
            command=command,
            args=list(spec.get("args") or []),
            env=env,
            cwd=spec.get("cwd") or None,
        )
        return "stdio", lambda: stdio_client(params)
    return None, None


# ---------------------------------------------------------------------------
# Async coroutines run on the portal's event loop
# ---------------------------------------------------------------------------

async def _list_all_tools(session: "ClientSession"):
    """Collect every tool across paginated tools/list pages (bounded so a buggy
    server that pages forever can't hang connect). Returns SDK `Tool` objects."""
    out = []
    cursor = None
    for _ in range(_MAX_LIST_PAGES):
        result = await session.list_tools(cursor) if cursor else await session.list_tools()
        out += [t for t in (result.tools or []) if getattr(t, "name", None)]
        cursor = result.nextCursor
        if not cursor:
            break
    return out


async def _run_conn(conn: "_Conn", build, connect_timeout: float):
    """Open one server's transport + ClientSession on the loop, run the handshake +
    tool listing under a connect timeout, then hold the scopes open until the stop
    event is set (so call_tool coroutines run against the live session). Any failure
    records conn.error and contributes no tools (graceful)."""
    try:
        async with build() as streams:
            read, write = streams[0], streams[1]
            async with ClientSession(read, write) as session:
                try:
                    with anyio.fail_after(connect_timeout):
                        await session.initialize()
                        tools = await _list_all_tools(session)
                except Exception as e:  # noqa: BLE001 — connect/handshake/list failure
                    conn.error = f"{type(e).__name__}: {e}"
                    log.warning("mcp: %s handshake/list failed: %s", conn.name, e)
                    conn._ready.set()
                    return
                conn.session = session
                conn.tools = tools
                conn._ready.set()
                log.info("mcp: %s connected (%d tool(s))", conn.name, len(tools))
                await conn._stop.wait()  # keep transport + session scopes open
    except Exception as e:  # noqa: BLE001 — transport-level failure (couldn't open)
        conn.error = f"{type(e).__name__}: {e}"
        log.warning("mcp: %s could not connect: %s", conn.name, e)
        conn._ready.set()
    finally:
        conn._done.set()


async def _invoke(session: "ClientSession", raw: str, arguments: dict, timeout: float):
    """Call one tool with the per-call timeout INSIDE the coroutine, so a hung server
    cancels this task (freeing the agent thread) instead of wedging the loop."""
    with anyio.fail_after(timeout):
        return await session.call_tool(raw, arguments or {})


def _render_result(res) -> str:
    """Flatten an SDK `CallToolResult` into text for the model. Concatenates text
    content blocks; notes non-text blocks (images/resources) by type. Honors isError
    by prefixing the text so the model treats it as a failure to react to."""
    parts = []
    for block in (getattr(res, "content", None) or []):
        btype = getattr(block, "type", None)
        if btype == "text":
            parts.append(str(getattr(block, "text", "")))
        elif btype == "resource":
            r = getattr(block, "resource", None)
            rtext = getattr(r, "text", None) if r is not None else None
            if rtext:
                parts.append(str(rtext))
            else:
                uri = getattr(r, "uri", "unknown") if r is not None else "unknown"
                parts.append(f"[resource: {uri}]")
        else:
            parts.append(f"[{btype or 'non-text'} content omitted]")
    text = "\n".join(p for p in parts if p)
    # Some servers return only structuredContent (no content blocks).
    sc = getattr(res, "structuredContent", None)
    if not text and sc is not None:
        try:
            text = json.dumps(sc, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            text = str(sc)
    if not text:
        text = "[no content]"
    if getattr(res, "isError", False):
        text = "[tool reported an error]\n" + text
    return text[:_RESULT_MAX_CHARS]


# ---------------------------------------------------------------------------
# One live server connection (bridged onto the registry's portal)
# ---------------------------------------------------------------------------

class _Conn:
    """One MCP server. Its transport + ClientSession live on the registry's portal
    loop inside `_run_conn`; this object is the synchronous handle the agent thread
    uses to call tools and to tear the connection down."""

    def __init__(self, name: str, spec: dict, transport: str):
        self.name = name
        self.spec = spec
        self.transport = transport       # "stdio" | "http"
        self.tools = []                  # SDK Tool objects (empty if connect failed)
        self.error: Optional[str] = None         # human string if the server is unusable
        self.session: Any = None         # live ClientSession (loop thread); set on connect
        self.portal: Any = None          # the registry's BlockingPortal
        self._stop: Any = None           # anyio.Event on the loop; set to unwind scopes
        self._ready = threading.Event()  # connect attempt finished (ok or error)
        self._done = threading.Event()   # connect task fully unwound

    def call(self, raw: str, arguments: dict) -> str:
        """Invoke one tool and return its result text. Raises on transport/timeout
        error; a tool-reported error (isError) is returned as text via _render_result."""
        timeout = self.spec.get("timeout")
        timeout = timeout if isinstance(timeout, (int, float)) else _CALL_TIMEOUT
        res = self.portal.call(partial(_invoke, self.session, raw, arguments, timeout))
        return _render_result(res)

    def close(self):
        """Signal the connect task to unwind its session + transport on the loop
        thread, then wait briefly for it so no subprocess / socket leaks."""
        if self.portal is not None and self._stop is not None:
            try:
                self.portal.call(self._stop.set)
            except Exception:  # noqa: BLE001 — portal may already be shutting down
                pass
        self._done.wait(timeout=5)


# ---------------------------------------------------------------------------
# Registry (cwd-cached, like skills._registry) — owns ONE portal for its lifetime
# ---------------------------------------------------------------------------

class _Registry:
    """All MCP servers configured for one cwd, connected once over a single shared
    portal loop. Holds the live connections plus a name -> (conn, tool) index for
    dispatch and a name -> schema/mutating index for validation."""

    def __init__(self, cwd: str):
        self.cwd = cwd
        self.clients = []                 # all _Conn (connected or errored)
        self.by_tool = {}                 # mcp__server__tool -> (conn, raw_tool_name)
        self._schemas = []                # OpenAI/Qwen schema per exposed tool
        self._param = {}                  # tool name -> JSON-Schema (for validate.py)
        self._mutating = set()            # tool names that need confirmation
        self.blocked = []                 # [(name, reason)] project servers gated as untrusted
        self.needs_login = []             # [(name, reason)] oauth servers awaiting /mcp login
        self.warnings = []
        self.portal = None
        self._portal_cm = None
        self._connect()

    def _connect(self):
        servers, warnings = _load_config(self.cwd)
        self.warnings = list(warnings)
        trusted = _is_trusted(self.cwd)

        # Decide which servers we'll actually connect (after disabled/name/trust gates).
        to_connect = []
        for name, spec, scope in servers:
            if spec.get("disabled"):
                log.info("mcp: %s disabled in config; skipping", name)
                continue
            if _SEP in name:
                self.warnings.append(f"{name}: server name contains '{_SEP}'; skipped")
                continue
            if scope == "project" and not trusted:
                # Never auto-prompt here: connection is lazy/mid-turn, where there's no
                # safe place to block on input. Default to not connecting + a warning;
                # the operator opts in out-of-band with `/mcp trust`.
                reason = "project server not started — project not trusted (run /mcp trust)"
                self.blocked.append((name, reason))
                self.warnings.append(f"{name}: {reason}")
                log.info("mcp: %s gated — project %s not trusted", name, self.cwd)
                continue
            if mcp_oauth.is_oauth(spec):
                # OAuth is opt-in and never auto-connects interactively. Flag off => skip
                # entirely (no tools, warning). Flag on but no stored tokens => "needs
                # login" (the operator runs /mcp login; we never block a turn/eval on a
                # browser). Flag on with tokens => connect non-interactively (SDK
                # uses/refreshes the stored token; a refresh failure degrades, never hangs).
                if not mcp_oauth.oauth_enabled():
                    reason = "OAuth disabled — set CHAD_MCP_OAUTH=1"
                    self.blocked.append((name, reason))
                    self.warnings.append(f"{name}: {reason}")
                    log.info("mcp: %s skipped — OAuth feature flag off", name)
                    continue
                url = spec.get("url")
                if not (isinstance(url, str) and url):
                    self.warnings.append(f"{name}: oauth server needs a 'url'")
                    continue
                if not mcp_oauth.has_tokens(name):
                    reason = f"needs login — run /mcp login {name}"
                    self.needs_login.append((name, reason))
                    self.warnings.append(f"{name}: {reason}")
                    log.info("mcp: %s needs interactive OAuth login", name)
                    continue
                storage = mcp_oauth.FileTokenStorage(name)
                provider = mcp_oauth.make_noninteractive_provider(url, storage, spec.get("scope"))
                kind, build = _transport_for(spec, auth=provider)
                to_connect.append((name, spec, kind, build))
                continue
            kind, build = _transport_for(spec)
            if build is None:
                self.warnings.append(f"{name}: no 'url' or 'command' in server config")
                continue
            to_connect.append((name, spec, kind, build))

        if not to_connect:
            return

        # Bring up the shared portal loop once, then launch every connect as a task so
        # they run in parallel — one dead/slow endpoint can't stall the others.
        self._portal_cm = start_blocking_portal()
        self.portal = self._portal_cm.__enter__()
        pending = []
        for name, spec, kind, build in to_connect:
            conn = _Conn(name, spec, kind)
            conn.portal = self.portal
            conn._stop = self.portal.call(anyio.Event)
            self.clients.append(conn)
            ct = spec.get("connect_timeout")
            ct = ct if isinstance(ct, (int, float)) else _CONNECT_TIMEOUT
            self.portal.start_task_soon(_run_conn, conn, build, ct)
            pending.append((conn, ct))

        # Join each connect, time-bounded. The grace lets the coroutine's own
        # fail_after report a clean error before our hard cutoff trips.
        for conn, ct in pending:
            if not conn._ready.wait(ct + _CONNECT_JOIN_GRACE):
                conn.error = conn.error or f"connect timed out after {ct}s"
            if conn.error:
                self.warnings.append(f"{conn.name}: {conn.error}")
            else:
                self._register(conn)

    def _register(self, conn: "_Conn"):
        for t in conn.tools:
            raw = t.name
            full = f"{PREFIX}{conn.name}{_SEP}{raw}"
            if full in self.by_tool:  # two servers, same tool name — keep first, warn
                self.warnings.append(f"{full}: duplicate tool name; later copy ignored")
                continue
            schema = t.inputSchema
            if not isinstance(schema, dict) or schema.get("type") != "object":
                schema = {"type": "object", "properties": {}}
            self.by_tool[full] = (conn, raw)
            self._param[full] = schema
            self._schemas.append({
                "type": "function",
                "function": {
                    "name": full,
                    "description": _describe(conn.name, t),
                    "parameters": schema,
                },
            })
            if _is_mutating(t):
                self._mutating.add(full)

    def schemas(self):
        return self._schemas

    def param_schema(self, name):
        return self._param.get(name)

    def is_mutating(self, name) -> bool:
        return name in self._mutating

    def call(self, name: str, arguments: dict) -> str:
        entry = self.by_tool.get(name)
        if entry is None:
            return f"[unknown MCP tool {name!r}]"
        conn, raw = entry
        try:
            return conn.call(raw, arguments)
        except Exception as e:  # noqa: BLE001 — surface transport/timeout errors to the model
            log.warning("mcp: call %s failed: %s", name, e)
            return f"[MCP error calling {name}: {type(e).__name__}: {e}]"

    def tool_names(self):
        return list(self.by_tool)

    def close(self):
        # Unwind every session on the loop thread first, then stop the loop itself.
        for c in self.clients:
            c.close()
        if self._portal_cm is not None:
            try:
                self._portal_cm.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
            self.portal = None
            self._portal_cm = None


def _describe(server: str, tool) -> str:
    """Tool description shown to the model, tagged with its origin server so the model
    (and the user reading a trace) knows it's an external MCP tool."""
    desc = (getattr(tool, "description", None) or "").strip()
    head = f"[MCP server '{server}'] "
    return head + desc if desc else head + tool.name


def _is_mutating(tool) -> bool:
    """Whether a tool needs the confirm gate. MCP `annotations.readOnlyHint == true`
    means the server promises no side effects -> safe to auto-run; anything else is
    treated as mutating (the safe default: an MCP tool may write files, hit an API, or
    send a message, and we'd rather over-confirm than act unprompted)."""
    ann = getattr(tool, "annotations", None)
    if ann is not None and getattr(ann, "readOnlyHint", None) is True:
        return False
    return True


_registry = None


def service() -> "_Registry":
    """The MCP registry for the current cwd, connected once and cached. Rebuilt if the
    working directory changed (each eval task / project switch gets its own servers)."""
    global _registry
    cwd = os.path.abspath(os.getcwd())
    if _registry is None or _registry.cwd != cwd:
        if _registry is not None:
            _registry.close()
        _registry = _Registry(cwd)
        if _registry.by_tool:
            log.info("mcp: %d tool(s) from %d server(s): %s",
                     len(_registry.by_tool), len(_registry.clients),
                     ", ".join(_registry.tool_names()))
    return _registry


def reset_session():
    """Tear down all server connections and force reconnection on next use. Called when
    a new Agent / `/reset` starts so a prior session's servers don't leak forward
    (mirrors skills.reset_session and lsp's rebind-stops-old-server)."""
    global _registry
    if _registry is not None:
        _registry.close()
    _registry = None


def trust(cwd: str = None):
    """Mark the current project (cwd) as trusted so its `./.mcp.json` servers connect,
    then reset the session so the next `service()` reconnects with them enabled. Wired
    to the `/mcp trust` command in both front-ends."""
    _set_trusted(cwd or os.getcwd())
    reset_session()


def login(name: str, emit=None) -> str:
    """Run the interactive OAuth flow for one configured `auth: oauth` server, persist
    the resulting tokens, and reset the session so the server's tools come live. Wired to
    `/mcp login <server>` in both front-ends. `emit` is an optional callable(str) the
    front-end passes so the browser URL / progress is shown in its own UI. Returns a
    human-readable status line and NEVER raises into the caller — a failed/abandoned
    login degrades to a clear message (the server simply stays unconnected).

    Why this is its own command (not auto-connect): the first OAuth connect opens a
    browser and blocks on a human approving, then catches a redirect. That cannot happen
    silently on an agent turn and must never wedge an eval/headless run."""
    say = emit or (lambda _m: None)

    if not mcp_oauth.oauth_enabled():
        return "OAuth is disabled — set CHAD_MCP_OAUTH=1 and retry."

    servers, _warnings = _load_config()
    spec = next((s for n, s, _scope in servers if n == name), None)
    if spec is None:
        return f"no MCP server named {name!r} in config."
    if not mcp_oauth.is_oauth(spec):
        return f"{name} is not an OAuth server (no \"auth\": \"oauth\" in its config)."
    url = spec.get("url")
    if not (isinstance(url, str) and url):
        return f"{name}: oauth server needs a 'url'."

    storage = mcp_oauth.FileTokenStorage(name)
    loopback = mcp_oauth.LoopbackServer()
    provider = mcp_oauth.make_login_provider(url, storage, loopback, spec.get("scope"), emit=say)
    headers = spec.get("headers")
    headers = headers if isinstance(headers, dict) else None

    async def _do_login():
        # initialize() triggers the SDK's OAuth flow (401 -> authorize -> browser ->
        # loopback redirect -> token exchange); tokens are persisted by `storage` along
        # the way. We only need the handshake to complete for that to happen.
        async with streamablehttp_client(url, headers=headers, auth=provider) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                with anyio.fail_after(mcp_oauth._LOGIN_TIMEOUT + 30):
                    await session.initialize()

    try:
        with start_blocking_portal() as portal:
            portal.call(_do_login)
    except Exception as e:  # noqa: BLE001 — any flow failure degrades to a message
        log.warning("mcp: %s OAuth login failed: %s", name, e)
        return f"{name}: login failed ({type(e).__name__}: {e})."
    finally:
        loopback.close()

    if not mcp_oauth.has_tokens(name):
        return f"{name}: login did not produce a token."
    reset_session()
    return f"{name}: logged in — its tools are now available."


# ---------------------------------------------------------------------------
# Thin module-level accessors (the surface tools.py / validate.py / agent.py use)
# ---------------------------------------------------------------------------

def schemas():
    """OpenAI/Qwen tool schemas for every connected MCP tool ([] if none)."""
    return service().schemas()


def is_mcp_tool(name: str) -> bool:
    return isinstance(name, str) and name.startswith(PREFIX)


def has_tool(name: str) -> bool:
    return name in service().by_tool


def param_schema(name: str):
    """JSON-Schema for an MCP tool's arguments, or None if not an MCP tool. Lets
    validate.py coerce/validate MCP calls with no schema duplication."""
    if not is_mcp_tool(name):
        return None
    return service().param_schema(name)


def is_mutating(name: str) -> bool:
    return is_mcp_tool(name) and service().is_mutating(name)


def call(name: str, arguments: dict) -> str:
    return service().call(name, arguments)


def summary_lines():
    """Human-readable rows for the `/mcp` command: one line per server (transport +
    tool count, or its connection error), each server's tools, then any warnings."""
    reg = service()
    if not reg.clients and not reg.blocked and not reg.needs_login:
        return ["no MCP servers configured. Add one to .mcp.json (project) or "
                "~/.chad/mcp.json (user): "
                '{"mcpServers": {"name": {"command": "...", "args": [...]}}} '
                'or {"name": {"type": "http", "url": "https://..."}}']
    out = []
    for c in reg.clients:
        if c.error:
            out.append(f"{c.name} [{c.transport}] — ✗ {c.error}")
            continue
        ro = sum(1 for t in c.tools if not _is_mutating(t))
        out.append(f"{c.name} [{c.transport}] — {len(c.tools)} tool(s) ({ro} read-only)")
        for t in c.tools:
            mark = "" if _is_mutating(t) else " (read-only)"
            desc = " ".join((getattr(t, "description", None) or "").split())
            if len(desc) > 70:
                desc = desc[:67] + "…"
            out.append(f"    {PREFIX}{c.name}{_SEP}{t.name}{mark}"
                       + (f" — {desc}" if desc else ""))
    for name, reason in reg.blocked:
        out.append(f"{name} — ⊘ blocked ({reason})")
    for name, reason in reg.needs_login:
        out.append(f"{name} [http/oauth] — ⊷ {reason}")
    out += warn_footer(reg.warnings)
    return out


@atexit.register
def _shutdown():
    if _registry is not None:
        _registry.close()
