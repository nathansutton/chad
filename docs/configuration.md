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
(seconds) bounds the initial connect. At startup chad connects the **eligible** servers
**in parallel and time-bounded** (one dead endpoint can't stall the others), runs the
`initialize` handshake, lists each server's tools (paginated), and registers them.
"Eligible" excludes servers that are gated: `"disabled": true` servers are skipped, OAuth
servers are deferred until you log in (see below), and — most importantly — **project-scope
`./.mcp.json` servers do not start until you `/mcp trust` the project** (next section).
User-scope `~/.chad/mcp.json` servers are authored by you and auto-connect.

**Project trust.** Dropping a `./.mcp.json` into a repo does **nothing** until you run
`/mcp trust` in that directory. This is deliberate: a project file is content you may have
just cloned, and a stdio server is an arbitrary local command — an untrusted repo must not
be able to auto-launch a subprocess the moment you open chad in it. Until trusted, its
servers show gated in `/mcp` (`project server not started — project not trusted (run /mcp
trust)`) and contribute no tools. `/mcp trust` records the project's absolute path in
`~/.chad/trusted_mcp.json` (mode `0600`); the path is the trust anchor, so moving the repo
to a new directory re-prompts. User-scope servers are exempt (you wrote them).

> **`CHAD_MCP_FULL_ENV`** — a stdio MCP subprocess inherits only a **minimal env allowlist**
> by default (`PATH`, `HOME`, `LANG`/locale, `TMPDIR`, `SHELL`, `USER`, … — enough to find
> its binary and start), **not** chad's full environment. A user-configured server runs an
> arbitrary local command and has no business inheriting your API keys, cloud tokens, or
> provider creds. A server that genuinely needs one var declares it in its config `env:`
> block (merged in, and it wins). Setting `CHAD_MCP_FULL_ENV=1` restores the full
> parent-environment inherit for the rare server that needs it — but it hands **every**
> secret in chad's environment to **every** stdio server, so leave it unset unless you know
> exactly which server needs what.

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
cap. `CHAD_MAX_CONTEXT` requests more and **YaRN-extends** a model past native when its
config supports it, capped at the model's documented max — so `CHAD_MAX_CONTEXT=262144`
resolves to "256k, or the model's max". KV cache grows lazily, so a large window costs
nothing until tokens fill it — and on the shipped Ornith models the KV cache is
**quantized to 8-bit by default** (half the fp16 footprint; `CHAD_KV_BITS=0`
restores fp16).

How much it costs depends on the model's attention design. The table below is
illustrative for a **pure-attention** transformer (e.g. the Qwen2.5-Coder models the
eval bench keeps for research), where the KV cache grows linearly with context:

| Context | KV cache (`CHAD_KV_BITS=0`, fp16) | KV cache (8-bit, default) |
|---|---|---|
| 32k | 1.2 GB | 0.6 GB |
| 128k (YaRN) | 4.8 GB | 2.4 GB |
| 256k | 9.7 GB | 4.8 GB |

Ornith — the model chad ships — is a **hybrid SSM/attention** model: its recurrent
layers carry a *fixed-size* state no matter how long the context gets, so only its
attention layers grow. Its real footprint is flatter than the table above and sits well
inside 24 GB alongside the ~5 GB of weights. When the prompt nears the window, old
verbose tool outputs are compacted.

8-bit KV used to cost ~20-30% throughput (mlx_lm's quantized attention is
unfused), which is why it was opt-in. chad now ships a fused quantized-KV
decode kernel (installed automatically when the model's attention shape is
covered — both shipped Ornith models are), making the quantized cache *faster*
than fp16 at long context (35B @32k: 60.2 vs 55.8 tok/s) on top of the RAM
halving, so it is the default. `CHAD_KV_BITS=0` restores the fp16 cache;
`CHAD_NO_QSDPA=1` keeps the quantized cache but disables the fused kernel
(debug only — that combination is the old slow path).

```bash
CHAD_MAX_CONTEXT=131072 uv run chad   # full 128k agentic context
```

## Advanced (env vars)

The rarely-touched tuning knobs live in environment variables so they stay off the main
`--help`. Same capability, sane defaults when unset:

```bash
CHAD_MAX_CONTEXT=131072 uv run chad      # YaRN-extend to the model's full 128k window
CHAD_KV_BITS=0          uv run chad      # fp16 KV cache (8-bit fused is the default where covered)
CHAD_CTX_LIMIT=28000    uv run chad      # force the compaction threshold (overrides the RAM-aware default)
CHAD_CTX_RESERVE_GB=2.5 uv run chad      # scratch RAM held back when auto-sizing that threshold
CHAD_CTX_SLOPE_FACTOR=1.0 uv run chad    # per-token cost multiplier for that auto-sizing (default
                                         # 1.75: measured peak grows ~1.75x the raw KV bytes/token)
CHAD_MODEL=/path/to/mlx-model uv run chad  # power-user escape hatch: run a different MLX model
CHAD_PREFILL_CHUNK=1024 uv run chad      # force a fixed prefill chunk (default: adaptive — MoE 2048
                                         # / dense 512, decaying to 256 as context+pressure grow)
CHAD_NO_MEMORY_CLAMP=1  uv run chad      # A/B knob: skip the Metal allocator clamps installed at load
```

By default the auto-compaction threshold (when chad reclaims old context — a full
re-prefill on this non-trimmable cache, so we do it as rarely as RAM allows) is **sized
automatically** from the live Metal memory budget and the model's measured per-token
cost, then capped at the model's window. The per-token cost is the KV bytes/token times
a slope factor (default 1.75) because measured peak memory grows faster than the KV
cache alone — prefill/decode scratch scales with resident context too. On a 24 GB Mac
running the 35B that lands around ~100k tokens. It self-calibrates per machine: less
RAM compacts sooner, more RAM runs nearer the full window. `CHAD_CTX_LIMIT` forces an
exact threshold (used by evals/tests); `CHAD_CTX_RESERVE_GB` (default 1.5) tunes how
much headroom is held back for prefill/decode scratch — raise it if you run other
memory-hungry apps alongside chad; `CHAD_CTX_SLOPE_FACTOR` tunes the per-token
multiplier (1.0 recovers the old raw-KV sizing).

`CHAD_MODEL` points chad at any local MLX model directory instead of Ornith. The harness
is tuned for Ornith, so this is unsupported and mostly there for research — the happy
path is the single bundled model, no flag.

**Memory safety (plan 075).** At load the engine wires the Metal working set and caps
the allocator slightly below it (`mx.set_wired_limit`/`set_memory_limit`), so a
transient spike back-pressures instead of escalating to a jetsam SIGKILL; a Metal OOM
caught inside a prefill chunk rolls the cache back exactly, halves the chunk, and
retries. `CHAD_NO_MEMORY_CLAMP=1` disables the clamps (A/B). The compaction threshold
additionally respects host-physical free memory (pressure from Docker or other apps
that the Metal budget can't see) and is re-checked between turns.

**`chad --serve`** exposes the in-process engine as an OpenAI-compatible endpoint
(`/v1/chat/completions`), keeping the prefix KV cache warm across requests — see
`benchmarks/tb2/README.md` for its role in the Terminal-Bench Mac arm. One request at
a time; `--host 0.0.0.0` to accept connections from Docker containers.

**Forcing the small model.** `CHAD_MODEL` also names another Ornith on Hugging Face, so
it doubles as the "use the 9B" recipe. chad's default keys on *physical* RAM, so a 24 GB
Mac with a lot already resident still gets the 35B (~14 GB peak) and can swap-thrash —
close other apps, or pin the 9B (~5 GB) for the session:

```bash
CHAD_MODEL=nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX uv run chad
```

### Alternate backend (spike)

The default backend is the in-process MLX engine — the whole point of chad (a persistent
prefix KV cache on-device). A research **spike** (`--backend openai`) drives the *same*
harness against any OpenAI-compatible `/v1/chat/completions` endpoint instead, to separate
harness-value from engine-value. It's unsupported and lossy (no local KV cache; honest
degradations apply) — not a general "use a cloud model" path.

```bash
uv run chad --backend openai --base-url http://localhost:8080/v1   # or CHAD_OPENAI_BASE_URL
```

- **`--base-url`** / **`CHAD_OPENAI_BASE_URL`** — the endpoint base URL; required for
  `--backend openai` (the flag wins; the env var is the fallback).
- **`--api-key-env NAME`** — the *name* of the env var holding the API key (read from that
  var, never passed on the command line). Omit for a local endpoint that needs no key.

### Turn budgets & think-cap

A runaway-turn **governor** ends a turn that burns a lot of prefill without landing and
verifying a change — it nudges at ~50% of budget and, at ~80%, banks a one-line progress
note and stops. On by default in one-shot/eval runs (interactively the human is the wall
clock). All three are also settable as flags (`--think-budget`, `--turn-budget-tokens`,
`--turn-budget-s`); the flag just sets the matching env knob.

```bash
CHAD_THINK_BUDGET=1500        uv run chad  # soft-cap each step's <think> at N tokens, then force-close + continue
CHAD_TURN_BUDGET_TOKENS=90000 uv run chad  # governor token budget (default 3× the context limit)
CHAD_TURN_BUDGET_S=600        uv run chad  # wall-clock variant (seconds); off by default
```

- **`CHAD_THINK_BUDGET`** — soft-caps each step's `<think>` run at N tokens, force-closes it,
  and continues (escalates when the model is stuck); off by default (plan 039).

  This is the **adaptive think-cap** — a smarter reasoning lever than the blunt `--no-think`:

  - **`--no-think`** kills **all** `<think>` for the whole session. It's the biggest,
    bluntest time saver (see [benchmarks](benchmarks.md) — "the most effective time-to-done
    lever on well-scoped agentic work"), but it costs pass-rate on hard tasks that genuinely
    need to reason. All-or-nothing.
  - **the adaptive think-cap** (`--think-budget N` / `CHAD_THINK_BUDGET=N`) leaves reasoning
    **on**, and only trims a step whose `<think>` run runs past N tokens — force-closing it
    (prefix-safe, so the next step is a cheap append, not a re-prefill) so reasoning can't
    balloon. The cap **escalates** with the turn's stuck-signals: a genuinely hard step that
    keeps getting capped is given more room instead of being chopped repeatedly. So it keeps
    full reasoning on well-scoped work and only trims the rambling. **Off by default today**
    — flipping it on by default is an eval-gated decision (see plan 057 / plan 039).

  When the cap fires during a turn, the TUI status line shows a small **`✂N`** counter (N =
  steps trimmed this turn) alongside the live ↑prefill / ↓generated readouts — so you can see
  it acting. With the cap off (the default) nothing renders.
- **`CHAD_TURN_BUDGET_TOKENS`** — the governor's cumulative-prefill budget per turn; defaults
  to 3× the context limit. Disable the governor entirely with `CHAD_NO_GOVERNOR=1` (below).
- **`CHAD_TURN_BUDGET_S`** — a wall-clock (seconds) variant of the same governor; off by
  default.

> **`CHAD_PREFILL_TRACE=path.jsonl`** is a dev/instrumentation knob, **not** supported
> config: it captures one JSON row per engine prefill to the given path for measurement
> spikes. Each row also carries the loop overhead outside the engine — `render_s`
> (chat-template re-tokenization), `compact_s`, and `prev_tools` (the prior step's tool
> executions as `[name, seconds]` pairs) — so a slow step can be attributed to prefill,
> tokenization, compaction, or a tool without guessing. Leave it unset in normal use.

### Harness levers & model profiles

A harness change that ships hardcoded can only be measured by reverting it. Behavioral
levers are therefore named and switchable, so a bundle of fixes can be attributed with
leave-one-out ablation instead of a revert per fix. `chad --levers` prints the registry
(it loads no model). A name that isn't registered is a startup error, not a warning: a
typo would otherwise run the unmodified harness and report the lever as having no effect.

Each lever carries a `group` (the harness iteration that introduced it, so one bundle can
be priced without paying for the others) and a `kind`:

- **`behavior`** — the change adds a behavior; without it the agent is merely less helped.
- **`regression-guard`** — the change fixes a demonstrated bug, and **OFF restores that
  bug**. A grep that reports `[no matches]` for a tree it never finished walking is a lie,
  not a configuration. These exist to be *measured*, never shipped off.

`tests/test_lever_bite.py` asserts every registered lever actually changes behavior when
disabled, so a "no measured effect" verdict from an ablation means the fix does nothing —
not that its guard was misplaced.

```bash
uv run chad --levers                              # inventory: every lever + what's active
CHAD_DISABLE=compact_notice uv run chad           # switch one behavior off
CHAD_DISABLE=plan_review,compact_offload uv run chad
CHAD_PROFILE=generic uv run chad                  # drop the Ornith-specific accommodations
CHAD_OFFLOAD_DIR=/tmp/off uv run chad             # where compaction spills untrimmed text
```

- **`CHAD_DISABLE`** — comma-separated lever names to turn off. All levers default **on**.
- **`CHAD_PROFILE`** — `ornith` (default) or `generic`. A profile block is strictly
  additive: the `<tool_call>` contract stays in the base prompt, so `generic` can still
  call tools. Resolved from the model id when unset, so an `--backend openai` run against
  a non-Ornith endpoint drops the accommodations automatically.
- **`CHAD_OFFLOAD_DIR`** — where the `compact_offload` lever writes the untrimmed
  transcript (default `~/.chad/offload`, capped at 32 MB, never inside the project).

### Safety & A/B opt-outs

These flip behavior off rather than tune it. The two safety opt-outs **weaken** chad's
defenses — leave them unset in normal use; they exist for measurement and edge cases.

```bash
CHAD_NO_SYMBOLS=1            uv run chad  # A/B knob: hide the tree-sitter symbolic tools
CHAD_NO_TASK=1              uv run chad  # A/B knob: hide the subagent/Task delegation tool
CHAD_NO_VALIDATE=1          uv run chad  # A/B knob: DISABLE arg coercion + schema validation
CHAD_NO_GOVERNOR=1          uv run chad  # A/B knob: DISABLE the runaway-turn governor
CHAD_NO_REPEAT_GUARD=1      uv run chad  # A/B knob: DISABLE the degenerate-repetition stop
CHAD_NO_SYNTAX_GATE=1       uv run chad  # A/B knob: DISABLE the post-edit syntax gate
CHAD_NO_PREFIX_CACHE=1      uv run chad  # measurement knob: drop the persistent prefix KV cache
CHAD_NO_DESTRUCTIVE_GUARD=1 uv run chad  # DISABLE the catastrophic-bash seatbelt (unsafe)
```

- **`CHAD_NO_SYMBOLS`** — drops the tree-sitter symbolic code-intel tools from the toolset
  (`tools.py`). A measurement knob used by the eval harness to A/B whether symbols help a
  given model; the plain bash/read/grep tools still work.
- **`CHAD_NO_TASK`** — hides the `task` tool (plan 041), which delegates open-ended
  exploration ("find where X happens") to a fresh **sub-agent** running in its own small,
  isolated context on a quarantined KV cache (`engine.push_cache`/`pop_cache`) and returns
  only a condensed answer — so the main transcript (and its prefill cost) stays small. The
  sub-agent is read-only by default and cannot spawn further sub-agents (depth 1). This
  knob is the A/B arm for measuring adoption/impact, and the escape hatch if a model
  misuses it.
- **`CHAD_NO_VALIDATE`** — **disables** the typia-style lenient-parse → typed-validate →
  self-repair loop for tool-call arguments (`validate.py`), falling back to a strict
  `json.loads` plus a terse missing-required check. This *weakens* input handling (malformed
  or loosely-typed tool calls that chad would normally coerce/repair will instead error). An
  A/B knob to measure what validation buys per model — leave unset in normal use.
- **`CHAD_NO_GOVERNOR`** — **disables** the runaway-turn governor (see [Turn budgets &
  think-cap](#turn-budgets--think-cap)), so a turn is never force-ended on its
  prefill/wall-clock budget. An A/B knob for measuring what the governor buys; the turn
  runs until the model stops on its own.
- **`CHAD_NO_REPEAT_GUARD`** — **disables** the degenerate-repetition stop (`guardrails.py`).
  Greedy decode on a small quantized model can lock into repeating one short string until
  the per-step token cap — minutes of dead generation per occurrence. By default chad
  watches the generation's tail, cuts the step off as soon as it turns fully periodic, and
  nudges the model out of the loop (aborting the turn after 3 cut-offs). Unlike the
  think-cap this never trades capability — it only fires on output that is already garbage
  — so it is on by default; this knob is the A/B arm.
- **`CHAD_NO_SYNTAX_GATE`** — **disables** the post-edit syntax gate (`syntaxgate.py`),
  which normally warns when an edit *introduces* a new syntax error (it never flags a
  pre-existing one). An A/B arm for `run_evals --ab`; leave unset in normal use.
- **`CHAD_NO_PREFIX_CACHE`** — a fairness/measurement knob that **drops** the persistent
  prefix KV cache (`engine.py`), forcing a full re-prefill every step. It exists to measure
  what the cache is worth and makes chad much slower — never set it in normal use.
- **`CHAD_NO_DESTRUCTIVE_GUARD`** — **disables** the catastrophic-bash seatbelt
  (`guardrails.py`) even in `--yolo`/auto mode. With it set, an injected `rm -rf ~`,
  `mkfs`, `dd of=/dev/…`, fork bomb, or `curl … | sh` is **not** screened before running.
  It is a seatbelt, not a security boundary (a sandbox is) — recommend leaving it unset.

### Symbolic code intel (repo map & language server)

Defaults are tuned for a big repo on a memory-tight machine (measured on an 11k-file
checkout); you rarely need to touch these.

```bash
CHAD_REPOMAP_WORKERS=4   uv run chad  # subprocess workers for a cold repo scan (1 = serial)
CHAD_LSP_TIMEOUT=10      uv run chad  # per-request language-server timeout, seconds (default 5)
CHAD_LSP_MAX_RSS_MB=2048 uv run chad  # recycle a language server past this process-tree RSS (default 1536)
```

- **`CHAD_REPOMAP_WORKERS`** — how many `python -c` subprocess workers a cold whole-repo
  tag scan shards across (`repomap.py`; default: cores−2, capped at 8). Workers import
  only `chad.repomap` — never the MLX engine. Tags persist per repo under
  `~/.chad/cache/repomap/` (mtime-validated per file), so warm sessions skip the scan.
- **`CHAD_LSP_TIMEOUT`** — deadline for each language-server request (`lsp.py`). On
  timeout the caller falls back to the tree-sitter backend, which labels its results
  NAME-MATCH ONLY rather than implying precision.
- **`CHAD_LSP_MAX_RSS_MB`** — after each request the server's process tree is measured;
  past this cap it is stopped and restarts fresh on the next request. Guards against an
  analysis server (pyright hit 4 GB on hot symbols) starving the GPU allocator that holds
  the model weights.

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
