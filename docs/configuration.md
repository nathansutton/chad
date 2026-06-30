# Configuration & reference

*Extending chad (Agent Skills, MCP servers) and the full flag/env-var reference. For the
basics, see the [README](../README.md).*

## Agent Skills (agentskills.io)

chad implements the open [Agent Skills](https://agentskills.io) format, so a skill
authored for Claude Code (or any compatible client) works here unchanged. A *skill* is a
folder with a `SKILL.md` — YAML frontmatter (`name` + `description`, plus optional
`license`/`compatibility`/`metadata`/`allowed-tools`) followed by markdown instructions —
optionally bundling `scripts/`, `references/`, and `assets/`.

**Where chad looks** (project skills override user skills on a name clash):

| Scope   | Paths |
| ------- | ----- |
| Project | `./.agents/skills/`, `./.claude/skills/` (relative to the working dir) |
| User    | `~/.agents/skills/`, `~/.claude/skills/` |

**Progressive disclosure** keeps context small (the point, on a local model):

1. **Catalog** — at startup, only every skill's `name` + `description` go into the system
   prompt as an `<available_skills>` block (~50-100 tokens each). `/skills` prints it.
2. **Activation** — when a task matches, the model calls the `activate_skill` tool (its
   `name` argument is enum-constrained to real skills, so it can't invent one). That loads
   *that one* skill's full instructions, wrapped in `<skill_content>`, with its bundled
   files listed.
3. **Resources** — referenced `scripts/`/`references/`/`assets/` files are read on demand
   with the normal `read` tool, against the skill's directory.

Parsing is lenient (a name that doesn't match its directory, an over-long field, or an
unquoted `colon: value` in YAML loads anyway, with a warning); only a missing description
or unparseable YAML is skipped. Activated instructions are exempt from context compaction
— durable guidance is never silently truncated — and re-activating a loaded skill is a
no-op. Implementation: `src/chad/skills.py` (discovery/parse/activate), with the catalog in
`prompt.py`, the tool in `tools.py`, and the compaction guard in `compaction.py`.

## MCP servers (modelcontextprotocol.io)

Agent Skills add *instructions*; **MCP** adds *tools*. chad can connect to external
[Model Context Protocol](https://modelcontextprotocol.io) servers — a GitHub server, a
Postgres server, a company's internal API server, or hosted connectors like Linear and
Slack — and expose their tools to the model alongside its builtins. chad uses the official
`mcp` SDK for transport, so it speaks both **stdio** (local subprocess servers) and
**Streamable HTTP** (hosted/remote servers).

**Configure** servers in either file (project overrides user on a name clash):

| Scope   | Path | Convention |
| ------- | ---- | ---------- |
| Project | `./.mcp.json` | the Claude-Code project convention |
| User    | `~/.chad/mcp.json` | applies to every project |

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
      "env": { "API_KEY": "..." }
    },
    "linear": {
      "type": "http",
      "url": "https://mcp.example.com/mcp",
      "headers": { "Authorization": "Bearer <token>" }
    }
  }
}
```

**Transport is chosen by which key is present:** `url` → Streamable HTTP; otherwise
`command` → stdio. The optional `type` (`"http"` / `"stdio"`) is accepted for clarity but
the `url`/`command` presence is authoritative. A stdio server's `command`/`args`/`env`/`cwd`
launches it; an HTTP server's `url` is reached over the network and any `headers` (e.g. a
static `Authorization: Bearer …` token) are sent on every request. `"disabled": true` skips
a server, `"timeout"` (seconds) overrides the per-call limit, and `"connect_timeout"`
(seconds) bounds the initial connect. At startup chad connects every server **in parallel
and time-bounded** (one dead endpoint can't stall the others), runs the `initialize`
handshake, lists each server's tools (paginated), and registers them.

> **Auth:** static bearer/PAT tokens via `headers` work out of the box. Hosted connectors
> that require **OAuth** (Linear, Slack, Atlassian, most SaaS) are supported behind an
> opt-in flag — see below.

**OAuth (hosted connectors).** Mark an HTTP server with `"auth": "oauth"` and set the
`CHAD_MCP_OAUTH=1` environment variable to enable it:

```json
{
  "mcpServers": {
    "linear": { "type": "http", "url": "https://mcp.linear.app/mcp", "auth": "oauth" }
  }
}
```

Because the first OAuth connect is **interactive** (it opens a browser and waits for you
to approve), OAuth servers do **not** auto-connect — that would hang an agent turn or an
eval. Instead they show as `needs login` in `/mcp`, and you authorize them explicitly:

```
/mcp login linear
```

This opens your browser (or prints the URL to paste, for a watched headless session),
catches the redirect on a one-shot `127.0.0.1` loopback server, exchanges the code for
tokens, and stores them in **`~/.chad/mcp_tokens.json` (mode 0600)**. After a successful
login the server's tools come live and reconnect non-interactively on later runs (the SDK
refreshes the token as needed). Notes:

- With `CHAD_MCP_OAUTH` unset, an `auth: oauth` server is skipped with a warning and the
  stdio/bearer/HTTP paths are unchanged — none of the OAuth code runs.
- Headless / no-browser sessions are **never blocked**: an OAuth server that can't complete
  an interactive login simply contributes no tools.
- Token values are never logged. The token file is created `0600` from the first write.

**How they behave in the harness:**

- **Namespaced** `mcp__<server>__<tool>`, so server tools can't collide with chad's
  builtins (`bash`/`read`/…) or with each other.
- **Same validation path** as builtins — each tool's `inputSchema` drives the typed-coerce
  + self-repair loop (`"3"`→`3`, missing-required detection), no schema duplication.
- **Confirmation gate.** A tool the server marks `readOnlyHint` runs without a prompt;
  every other MCP tool is treated as mutating (the safe default — it might write files, hit
  an API, or send a message), so it asks first and shows its arguments. In `--plan` mode
  only read-only MCP tools run.
- **Graceful degradation.** A server that's missing, misconfigured, slow to connect, or
  that crashes mid-session contributes no tools and never takes the agent down. `/mcp` shows
  each server's transport (stdio/http), status, tools (with read-only markers), and warnings.

Implementation: `src/chad/mcp.py` drives the official `mcp` SDK behind a synchronous,
cwd-keyed registry — the SDK's async event loop runs in a background thread (one
`anyio` BlockingPortal) and every call is marshalled onto it, with the per-call timeout
inside the coroutine so a hung server can never wedge the agent. Wired into
`tools.active_schemas`/`dispatch_for`/`is_mutating`, the validator (`validate.py`), and the
agent loop (`agent.py`).

## Context window (agentic coding needs room)

By default the harness uses the model's **full native window** instead of an arbitrary
cap. `--max-context` requests more and **YaRN-extends** a model past native when its
config supports it, capped at the model's documented max — so `--max-context 262144`
resolves to "256k, or the model's max". KV cache grows lazily, so a large window costs
nothing until tokens fill it, and `--kv-bits 8` halves its footprint.

How much it costs depends on the model's attention design. The table below is
illustrative for a **pure-attention** transformer (e.g. the Qwen2.5-Coder models the
eval bench keeps for research), where the KV cache grows linearly with context:

| Context | KV cache (fp16) | KV cache (`--kv-bits 8`) |
|---|---|---|
| 32k | 1.2 GB | 0.6 GB |
| 128k (YaRN) | 4.8 GB | 2.4 GB |
| 256k | 9.7 GB | 4.8 GB |

Ornith — the model chad ships — is a **hybrid SSM/attention** model: its recurrent
layers carry a *fixed-size* state no matter how long the context gets, so only its
attention layers grow. Its real footprint is flatter than the table above and sits well
inside 24 GB alongside the ~5 GB of weights. When the prompt nears the window, old
verbose tool outputs are compacted. Note `CHAD_KV_BITS=8` costs ~20–30%
throughput (dequant overhead), so leave it off unless you actually need the memory.

```bash
CHAD_MAX_CONTEXT=131072 CHAD_KV_BITS=8 uv run chad   # full 128k agentic context
```

## Advanced (env vars)

The rarely-touched tuning knobs live in environment variables so they stay off the main
`--help`. Same capability, sane defaults when unset:

```bash
CHAD_MAX_CONTEXT=131072 uv run chad      # YaRN-extend to the model's full 128k window
CHAD_KV_BITS=8          uv run chad      # quantize the KV cache (~half the RAM, ~20-30% slower)
CHAD_CTX_LIMIT=28000    uv run chad      # force the compaction threshold (overrides the RAM-aware default)
CHAD_CTX_RESERVE_GB=2.5 uv run chad      # scratch RAM held back when auto-sizing that threshold
CHAD_MODEL=/path/to/mlx-model uv run chad  # power-user escape hatch: run a different MLX model
```

By default the auto-compaction threshold (when chad reclaims old context — a full
re-prefill on this non-trimmable cache, so we do it as rarely as RAM allows) is **sized
automatically** from the live Metal memory budget and the model's measured per-token KV
cost, then capped at the model's window. On a 24 GB Mac running the 35B that lands around
~175k tokens (vs a fixed 120k before) — long sessions stay warm and TTFT flat. It
self-calibrates per machine: less RAM compacts sooner, more RAM runs nearer the full
window. `CHAD_CTX_LIMIT` forces an exact threshold (used by evals/tests);
`CHAD_CTX_RESERVE_GB` (default 1.5) tunes how much headroom is held back for
prefill/decode scratch — raise it if you run other memory-hungry apps alongside chad.

`CHAD_MODEL` points chad at any local MLX model directory instead of Ornith. The harness
is tuned for Ornith, so this is unsupported and mostly there for research — the happy
path is the single bundled model, no flag.

### Safety & A/B opt-outs

These flip behavior off rather than tune it. The two safety opt-outs **weaken** chad's
defenses — leave them unset in normal use; they exist for measurement and edge cases.

```bash
CHAD_NO_SYMBOLS=1            uv run chad  # A/B knob: hide the tree-sitter symbolic tools
CHAD_NO_VALIDATE=1          uv run chad  # A/B knob: DISABLE arg coercion + schema validation
CHAD_NO_DESTRUCTIVE_GUARD=1 uv run chad  # DISABLE the catastrophic-bash seatbelt (unsafe)
```

- **`CHAD_NO_SYMBOLS`** — drops the tree-sitter symbolic code-intel tools from the toolset
  (`tools.py`). A measurement knob used by the eval harness to A/B whether symbols help a
  given model; the plain bash/read/grep tools still work.
- **`CHAD_NO_VALIDATE`** — **disables** the typia-style lenient-parse → typed-validate →
  self-repair loop for tool-call arguments (`validate.py`), falling back to a strict
  `json.loads` plus a terse missing-required check. This *weakens* input handling (malformed
  or loosely-typed tool calls that chad would normally coerce/repair will instead error). An
  A/B knob to measure what validation buys per model — leave unset in normal use.
- **`CHAD_NO_DESTRUCTIVE_GUARD`** — **disables** the catastrophic-bash seatbelt
  (`guardrails.py`) even in `--yolo`/auto mode. With it set, an injected `rm -rf ~`,
  `mkfs`, `dd of=/dev/…`, fork bomb, or `curl … | sh` is **not** screened before running.
  It is a seatbelt, not a security boundary (a sandbox is) — recommend leaving it unset.

### Session log & privacy

Diagnostics log: each session appends throughput numbers and a readable trace —
the user query, tool-call args (including bash commands and write/edit content),
and result previews — to `~/.chad/session.log`. It's now size-bounded (rotated,
5 MB × 3) and passes previews through a best-effort secret redactor, but it still
records command/file previews in plaintext outside the repo, so treat it as
sensitive. Set **`CHAD_NO_SESSION_LOG=1`** (any truthy value) to disable the session
log entirely — chad installs a null handler and won't create `~/.chad` for the log's
sake. (For the same privacy reason, the resumable conversation store under
`~/.chad/sessions/` — which holds full tool args and results — is created mode `0600`.)
