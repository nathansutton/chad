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
   prompt as an `<available_skills>` block, one `- name: description` line each (~40-80
   tokens). `/skills` prints it.
2. **Activation** — when a task matches, the model calls the `activate_skill` tool (its
   `name` argument is enum-constrained to real skills, so it can't invent one). That loads
   *that one* skill's full instructions, wrapped in `<skill_content>`, with its bundled
   files listed.
3. **Resources** — referenced `scripts/`/`references/`/`assets/` files are read on demand
   against the skill's directory, from `bash` (`sed -n '1,120p' <file>`).

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
  builtins (`bash`/`edit`/…) or with each other.
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
nothing until tokens fill it — and on the shipped model the KV cache is
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

Qwen3.8-27B — the model chad ships — is a **hybrid SSM/attention** model: 48 of its 64
layers are GatedDeltaNet and carry a *fixed-size* recurrent state no matter how long the
context gets, so only the 16 full-attention layers grow. Its real footprint is much flatter
than the table above (measured: 34,816 bytes per token). When the prompt nears the window,
old verbose tool outputs are compacted.

8-bit KV used to cost ~20-30% throughput (mlx_lm's quantized attention is
unfused), which is why it was opt-in. chad now ships a fused quantized-KV
decode kernel (installed automatically when the model's attention shape is
covered — the shipped model is), making the quantized cache *faster*
than fp16 at long context (measured on the retired 35B at 32k: 60.2 vs 55.8 tok/s) on top
of the RAM halving, so it is the default. `CHAD_KV_BITS=0` restores the fp16 cache;
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
CHAD_CTX_SAFETY=0.95    uv run chad      # the single headroom lever: fraction of the Metal budget
                                         # the auto-sizing may spend (default 0.975 — hold back 2.5%)
CHAD_CTX_SLOPE_FACTOR=1.5 uv run chad    # A/B knob: per-token cost multiplier for that auto-sizing
                                         # (default 1.0 — a token's marginal cost is its KV cost)
CHAD_MODEL=/path/to/mlx-model uv run chad  # power-user escape hatch: run a different MLX model
                                         # (also: --model auto|<repo> — the CLI twin, wins over this)
CHAD_PREFILL_CHUNK=1024 uv run chad      # force a fixed prefill chunk (default: adaptive — MoE 2048
                                         # / dense 512, decaying to 256 as context+pressure grow)
CHAD_NO_MEMORY_CLAMP=1  uv run chad      # A/B knob: skip the Metal allocator clamps installed at load
```

By default the auto-compaction threshold (when chad reclaims old context — a full
re-prefill on this non-trimmable cache, so we do it as rarely as RAM allows) is **sized
automatically** from the live Metal memory budget and the model's measured per-token
cost, then capped at the model's window. Three things are subtracted before the division:
the resident weights, a headroom band (`CHAD_CTX_SAFETY`), and the **prefill transient** —
the attention scratch that is live at the same moment the cache is. That last one is fixed,
not per-token: it climbs with context and then flattens once the adaptive chunker starts
shrinking the chunk (measured: 1.8 GB at 8k, 4.15 GB at 49k, flat thereafter), and past
that point peak memory grows at exactly the KV rate. On a 24 GB Mac the model's 12.3 GB of
weights plus that 4.3 GB transient spend 87% of the budget before the first cached token,
which is why it lands near ~56k rather than its 262k native window. It
self-calibrates per machine: less RAM compacts sooner, more RAM runs nearer the full
window. `CHAD_CTX_LIMIT` forces an
exact threshold (used by tests); `CHAD_CTX_SAFETY` (default 0.975) is the single
headroom lever — the fraction of the Metal budget the sizing may spend, so lower it if
you run other memory-hungry apps alongside chad; `CHAD_CTX_SLOPE_FACTOR` tunes the
per-token multiplier (1.0 recovers the raw-KV sizing).

The banner states the window you will **actually get**, not the checkpoint's native one.
On a memory-tight box the two differ by more than 2x — a 262k model on a 24 GB Mac gets
tens of thousands of tokens once the weights and the KV cache are paid for — and the
native number is context the run can never spend. When the governor is what bound it,
the banner says so: `84k of 262k context`.

`--model` (or `CHAD_MODEL`) takes `auto` — the shipped model — or any Hugging Face repo id
/ local MLX model directory. There are no size shorthands: 2.0.0 retired the Ornith 35B/9B
pair and the RAM-aware pick that chose between them, so `--model 9b` is now just a literal
(and nonexistent) repo id rather than a silent alias. Pointing it at other weights is
supported and kept on purpose — one shipped model is a default, not a restriction — but the
harness is *tuned* to the shipped model, so the realistic cost is throughput, not
correctness. The flag wins over `CHAD_MODEL`, so a globally exported var can't pin every
run.

**Memory safety.** At load the engine wires the Metal working set and caps
the allocator slightly below it (`mx.set_wired_limit`/`set_memory_limit`), so a
transient spike back-pressures instead of escalating to a jetsam SIGKILL; a Metal OOM
caught inside a prefill chunk rolls the cache back exactly, halves the chunk, and
retries. `CHAD_NO_MEMORY_CLAMP=1` disables the clamps (A/B). The compaction threshold
additionally respects host-physical free memory (pressure from Docker or other apps
that the Metal budget can't see) and is re-checked between turns.

### The model

chad ships exactly one: [`nathansutton/Qwen3.8-27B-UD-Q3_K_XL-MTP-MLX`][model], a dense
`qwen3_5` hybrid (64 layers — 48 GatedDeltaNet, 16 full attention) quantized to 3-bit
group-64 with `lm_head` held at 5-bit. ~12 GB resident, 262k native context, and it carries
the checkpoint's own trained MTP head as `mtp.safetensors` for
[self-speculative decoding](#speculative-decoding--kernel-knobs).

[model]: https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-MTP-MLX

**chad targets 24 GB Apple Silicon and nothing smaller.** There is no low-RAM fallback any
more, because there is no second model. On a smaller box chad prints a one-line warning at
startup and runs anyway — it advises, it does not gate — but ~12 GB of weights plus the
~4.3 GB prefill transient leave almost nothing for a KV cache, so the window collapses
toward its floor. Even at 24 GB the honest figure is roughly ~56k of the model's 262k
window; the banner states what you actually got, and the
[context window](#context-window-agentic-coding-needs-room) section explains the sizing.

Where the bits went, since a 3-bit model invites the question. On a dense model every
parameter is on the critical path for every token, so there is no expert redundancy to
absorb quantization error and no free lunch — but shrinking the weights is also the only
decode lever available. The recipe follows what the calibrated GGUF builds of this same
checkpoint agree on: `lm_head` is a second full 1.27B-param tensor (vocab 248,320,
`tie_word_embeddings` false) and is where protection is worth buying, while `embed_tokens`
is a lookup table whose per-row error never compounds through a matmul and is the cheapest
tier to cut. Holding both high, as a uniform "sensitive tier" would, spends ~0.78 GB on the
tier the evidence says needs it least — and on a 24 GB box 1 GB of weights is about 16k
tokens of context.

To run different weights through the same engine:

```bash
uv run chad --model mlx-community/Some-Other-Model-4bit   # any HF repo id
uv run chad --model /path/to/local/mlx-model              # or a local dir
CHAD_MODEL=/path/to/local/mlx-model uv run chad           # env equivalent
```

Precedence is `--model` → `CHAD_MODEL` → the shipped default. Both are honored without
argument; nothing about the harness is tuned for them.

### Alternate backend (remote)

The default backend is the in-process MLX engine — the whole point of chad (a persistent
prefix KV cache on-device). `--backend llama` instead drives the *same* harness against a
remote **llama.cpp** server's raw `/completion` endpoint (token-id prompts). This is the
arm used when chad runs inside a Linux benchmark container against a GGUF served on a GPU
box, where MLX can't run. It's lossy relative to the in-process engine (the KV cache lives
in the server, so warm-prefix checkpoints and cache-quarantine are no-ops) but keeps real
cache telemetry and passes `<think>` back verbatim — not a general "use a cloud model" path.

```bash
uv run chad --backend llama --base-url http://<host>:8081   # or CHAD_LLAMA_BASE_URL
```

- **`--base-url`** / **`CHAD_LLAMA_BASE_URL`** — the llama-server origin (bare, no `/v1`);
  required for `--backend llama` (the flag wins; the env var is the fallback).
- **`--tokenizer`** / **`CHAD_TOKENIZER`** — HF repo whose tokenizer matches the served
  model's vocab; required for a GGUF server (GGUF repos ship no tokenizer).
- **`--api-key-env NAME`** — the *name* of the env var holding the API key (read from that
  var, never passed on the command line). Omit for a local endpoint that needs no key.

### Serving the local model to a container (`chad serve`)

`--backend llama` exists because a Linux container can't run MLX. The usual answer is to
point it at a llama.cpp server holding a GGUF — but then the thing being measured is a
different quantization of the model than the one people actually run. `chad serve` closes
that gap: it speaks the *same* `/completion` protocol, backed by the local MLX engine and
its real prefix cache, so the container drives **this** machine's model unchanged.

```bash
uv run chad serve --host 0.0.0.0 --port 8081     # on the Mac holding the weights
# then, from the container (or another host):
chad "…" --backend llama --base-url http://host.docker.internal:8081
```

- **`--host`** / **`CHAD_SERVE_HOST`** — bind address. Defaults to `127.0.0.1`; a container
  reaching in over `host.docker.internal`, or any other machine, needs `0.0.0.0`.
- **`--port`** / **`CHAD_SERVE_PORT`** — TCP port (default `8081`).
- **`CHAD_SERVE_API_KEY`** — require `Authorization: Bearer <key>`. There is no auth by
  default, which is why the default bind is loopback; set this whenever you widen it, and
  give the client `--api-key-env`.

The engine knobs are the same ones a local `chad` reads, and they mean the same thing
here — the server is the local product, so `CHAD_MAX_CONTEXT`, `CHAD_KV_BITS`,
`CHAD_KV_CACHE_MAX_GB` and the full sampler family (`CHAD_TEMP`, `CHAD_MIN_P`,
`CHAD_TOP_P`, `CHAD_TOP_K`, `CHAD_PRESENCE_PENALTY`) all apply to the model it serves.
(In 1.x the server read *none* of the sampler vars — a `chad serve` started with
`CHAD_MIN_P` set ran without it and said nothing. They travel as one call now, so the
server and a local run cannot drift apart one setting at a time.) A knob a request doesn't mention keeps the server's value; a request that sends
one explicitly wins for that request only, so a client can A/B a sampler setting against a
server without restarting it.

`GET /props` reports the context window the server actually enforces, and clients should
budget against it. That number is pinned at load and never moves, because clients read it
once and size everything else against it. A prompt that doesn't fit is refused with `400`
before anything is prefilled, and a `n_predict` larger than the room left beside the
prompt is clamped to fit — overrunning the window is a Metal allocation the engine may not
survive, and one client's bad budget shouldn't take down everyone else's session.

The wall requests are admitted against is **live**, and can be tighter than the advertised
window. The Metal budget is blind to other processes, so a container stack or a browser
started after the server took physical pages the KV cache needs one-for-one; a prompt that
fit at load may not fit now. `GET /health` reports both — `n_ctx` (advertised) and
`safe_ctx` (what fits right now), plus `ctx_pressure` when they diverge — so you can see a
tightened wall coming instead of meeting it as a `400`. `CHAD_CTX_SAFETY` tunes the
headroom the estimate holds back (default 0.975).

Two things it does that a stock llama.cpp server can't, because both ends are chad's — it
advertises them in `/props` and the client feature-detects, so nothing changes when you
point the same client at a real llama-server:

- **cache quarantine** — a client can bracket an excursion with a real engine
  push/pop, so it does not evict the main transcript's prefix.
- **warm prefix** — the on-disk KV warm-start of the stable system+tools prefix, which a
  remote client can't do because the checkpoint lives on the server's disk.

The engine holds **one** KV cache, so generation is serialized: one agent at a time.
Concurrent clients queue rather than thrash the prefix (`/props` and `/health` stay
lock-free so a monitor can poll during a long turn). Two caveats worth stating before you
trust a number that comes out of this: scores are *not* comparable to a GGUF run (different
quantization is the whole point of the exercise), and on a laptop the agent's own container
workload competes with decode — on a wall-clock-timed benchmark, a task can fail on time
rather than on capability. Pilot a handful of tasks before trusting a full sweep.

### Sampling & reasoning effort

chad decodes **greedily by default** (temperature 0): reproducible, and the cheapest path
through the engine. That default has one field-measured failure mode worth knowing — a
stalled or garbled step replays itself byte-identically on every retry, and across
"independent" benchmark reps. Unattended harnesses should set a temperature so a retry can
take a different path.

```bash
CHAD_TEMP=0.7             uv run chad  # sampling temperature (default 0 = greedy)
CHAD_MIN_P=0.05           uv run chad  # min-p tail trim (default 0 = off)
CHAD_TOP_P=0.95           uv run chad  # nucleus sampling (default 0 = off)
CHAD_TOP_K=20             uv run chad  # top-k tail trim (default 0 = off)
CHAD_PRESENCE_PENALTY=0.5 uv run chad  # flat penalty on already-generated tokens (default 0)
CHAD_REASONING_EFFORT=low uv run chad  # template-level reasoning budget, where supported
```

- **`CHAD_MIN_P` / `CHAD_TOP_P` / `CHAD_TOP_K`** — anti-confabulation knobs for a heavily
  quantized model, all off by default. They trim the sub-noise-floor logit tail without
  touching temperature, which is usually what you want when a small quant invents an API.
- **`CHAD_PRESENCE_PENALTY`** — a flat score penalty on every already-emitted token.
  Ships at **0.0** and is worth leaving there, even though model cards suggest up to 1.5:
  those ranges are written for chat, and code is inherently repetitive — identifiers,
  keywords and punctuation *must* be reused. Measured on one task at 1.5, the model spent
  45 steps in pure exploration, landed zero edits, and emitted visibly corrupted tool
  arguments. Treat it as a knob to probe, not a default to ship.
- **`CHAD_REASONING_EFFORT`** — `xhigh` | `medium` | `low`, passed to the chat template as
  a reasoning budget on checkpoints whose template accepts one (Qwen3.8). Unset, the
  argument is not passed *at all*, so a template without the knob is unaffected. This is a
  template-level request to the model, distinct from the harness-level
  [think-cap](#turn-budgets--think-cap) below, which force-closes a `<think>` run the model
  has already started.

All five sampler settings are applied as one call, so every path that builds an engine —
interactive, one-shot, and `chad serve` — honors the same set.

### Turn budgets & think-cap

A runaway-turn **governor** ends a turn that burns a lot of prefill without landing and
verifying a change — it nudges at ~50% of budget and, at ~80%, banks a one-line progress
note and stops. On by default in one-shot/eval runs (interactively the human is the wall
clock).

These are env-only knobs because the only thing that sets them is an unattended harness,
which already builds a `CHAD_*` environment. `--think-budget` is the one member of the
family kept on the CLI, since it is a capability/latency trade a person might reach for
interactively; the former `--think-ceiling`, `--turn-budget-tokens`, `--turn-budget-s`,
`--auto-continue` and `--review-pass` flags are gone, and passing one is now an error
rather than a silent no-op.

```bash
CHAD_THINK_BUDGET=1500        uv run chad  # soft-cap each step's <think> at N tokens, then force-close + continue
CHAD_THINK_CEILING=384        uv run chad  # force-close a runaway <think> but keep decoding the action in the SAME step (off by default)
CHAD_TURN_BUDGET_TOKENS=90000 uv run chad  # governor token budget (default 3× the context limit)
CHAD_TURN_BUDGET_S=600        uv run chad  # wall-clock variant (seconds); off by default
CHAD_AUTO_CONTINUE=2          uv run chad  # on a hard stop, relaunch a fresh turn seeded with the progress note, N times
CHAD_REVIEW_PASS=1            uv run chad  # if a one-shot finishes early and clean, spend the slack verifying it
CHAD_MAX_GEN_TOKENS=32768     uv run chad  # hard per-STEP generation cap (default 32768)
```

- **`CHAD_THINK_BUDGET`** — soft-caps each step's `<think>` run at N tokens, force-closes it,
  and continues (escalates when the model is stuck); off by default.

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
    — flipping it on by default is an eval-gated decision.

  When the cap fires during a turn, the TUI status line shows a small **`✂N`** counter (N =
  steps trimmed this turn) alongside the live ↑prefill / ↓generated readouts — so you can see
  it acting. With the cap off (the default) nothing renders.
- **`CHAD_THINK_CEILING`** — the **close-and-continue** ceiling, and the one to reach for
  before `CHAD_THINK_BUDGET`. Where the think-cap force-closes `<think>` and *ends* the
  step (so the model re-derives its reasoning next step), this force-closes the runaway
  block and **keeps decoding the action in the same step** — the reasoning so far stays
  in context and nothing is re-derived. **Off by default**: force-closing `</think>`
  mid-generation is the most invasive thing the harness can do to the token stream, and
  the measured record says the bare loop doesn't need it. Steps it fires on are counted
  as *salvaged* in the session log.
- **`CHAD_TURN_BUDGET_TOKENS`** — the governor's cumulative-prefill budget per turn; defaults
  to 3× the context limit. Disable the governor entirely with `CHAD_NO_GOVERNOR=1` (below).
- **`CHAD_TURN_BUDGET_S`** — a wall-clock (seconds) variant of the same governor; off by
  default.
- **`CHAD_MAX_GEN_TOKENS`** — the hard ceiling on a single *step's* generation, 32768 by
  default. It is a backstop against non-repetitive runaway garble, not a reasoning lever:
  a literal decode loop is the repeat guard's job. It was 8192 through 1.12; raising it
  fixed a real loss class where a long chain of thought pinned the cap while still inside
  `<think>`, ending the step as discarded reasoning with no action.

> **`CHAD_PREFILL_TRACE=path.jsonl`** is a dev/instrumentation knob, **not** supported
> config: it captures one JSON row per engine prefill to the given path for measurement
> spikes. Each row also carries the loop overhead outside the engine — `render_s`
> (chat-template re-tokenization), `compact_s`, and `prev_tools` (the prior step's tool
> executions as `[name, seconds]` pairs) — so a slow step can be attributed to prefill,
> tokenization, compaction, or a tool without guessing. Leave it unset in normal use.

### Harness levers

chad 2.0.0 ships **eight** result-channel levers, **all ON by default** — the survivors
of a long measurement campaign in which nothing else beat the bare model + tool loop.
Each one makes the `bash` route more honest or more informative (a trimmed test run
keeps its failure rows; a grep that matched nothing says what it searched; a failed
edit explains itself), and each keeps a name and a switch for exactly one reason:
leave-one-out ablation. `chad levers` prints the registry (it loads no model). A name
that isn't registered is a startup error, not a warning: a typo would otherwise run
the unmodified harness and report the lever as having no effect.

```bash
uv run chad levers                          # inventory: every lever + what's active
CHAD_DISABLE=bash_line_clip uv run chad     # leave-one-out ablation arm
CHAD_DISABLE=all uv run chad                # the bare model + tool loop
```

- **`CHAD_DISABLE`** — comma-separated lever names to switch off (`all` = every lever).

### Safety & A/B opt-outs

These flip behavior off rather than tune it. The two safety opt-outs **weaken** chad's
defenses — leave them unset in normal use; they exist for measurement and edge cases.

```bash
CHAD_NO_VALIDATE=1          uv run chad  # A/B knob: DISABLE arg coercion + schema validation
CHAD_NO_GOVERNOR=1          uv run chad  # A/B knob: DISABLE the runaway-turn governor
CHAD_NO_REPEAT_GUARD=1      uv run chad  # A/B knob: DISABLE the degenerate-repetition stop
CHAD_NO_SYNTAX_GATE=1       uv run chad  # A/B knob: DISABLE the post-edit syntax gate
CHAD_NO_PREFIX_CACHE=1      uv run chad  # measurement knob: drop the persistent prefix KV cache
CHAD_NO_SKILLS=1            uv run chad  # A/B knob: disable all Agent Skill discovery
CHAD_NO_FASTPATH=1          uv run chad  # A/B knob: disable the fused-projection decode fast path
CHAD_NO_DESTRUCTIVE_GUARD=1 uv run chad  # DISABLE the catastrophic-bash screen (unsafe)
CHAD_NO_SEATBELT=1          uv run chad  # DISABLE the macOS Seatbelt sandbox for yolo bash (unsafe)
CHAD_NO_ENV_GUARD=1         uv run chad  # let bash children inherit credential-shaped env vars
CHAD_PROTECT_GIT=1          uv run chad  # also write-DENY .git inside the yolo sandbox
```

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
  pre-existing one). An A/B arm for evaluation runs; leave unset in normal use.
- **`CHAD_NO_PREFIX_CACHE`** — a fairness/measurement knob that **drops** the persistent
  prefix KV cache (`engine.py`), forcing a full re-prefill every step. It exists to measure
  what the cache is worth and makes chad much slower — never set it in normal use.
- **`CHAD_NO_DESTRUCTIVE_GUARD`** — **disables** the catastrophic-bash screen
  (`guardrails.py`) even in `--yolo` mode. With it set, an injected `rm -rf ~`,
  `mkfs`, `dd of=/dev/…`, fork bomb, or `curl … | sh` is **not** screened before running.
  It is a screen, not a security boundary (the sandbox below is) — leave it unset.
- **`CHAD_NO_SEATBELT`** — **disables** the macOS Seatbelt sandbox (`seatbelt.py`) that
  yolo-mode bash commands run under by default: file writes confined to the workspace,
  temp dirs, and caches; reads and network open. Only the spawned shell child is ever
  sandboxed. Set this only when the sandbox itself breaks a legitimate workflow.
- **`CHAD_NO_ENV_GUARD`** — bash children normally get a **filtered** copy of the
  environment: variable names shaped like credentials (`…_TOKEN`, `…_SECRET`,
  `…_API_KEY`, …) are dropped — name-pattern only, values never read. Set this for a
  session whose commands legitimately need a credential (e.g. `gh`, deploy scripts).
- **`CHAD_PROTECT_GIT`** — an opt-in tier on top of the yolo sandbox: the workspace's
  `.git` (and a worktree's external gitdir) is write-DENIED, so an unreviewed command
  cannot destroy project history. The cost is real — every `.git`-writing git command
  (commit, add, checkout) EPERMs inside the sandbox — which is why it is opt-in.
- **`CHAD_NO_SKILLS`** — turns off [Agent Skill](#agent-skills-agentskillsio) discovery
  entirely: no `# Skills` prompt section, no skill tool. Set it when a benchmark must not
  inherit your personal skills, or to A/B what skills are worth. Unlike the other
  `CHAD_NO_*` vars this one wants a real truthy value (`1`/`true`/`yes`/`on`).
- **`CHAD_NO_FASTPATH`** — disables the fused-projection + compiled decode step installed
  at load for the hybrid MoE checkpoint (`mlx_fastpath.py`). Pure speed, no behavior
  change, so this is an A/B and bisection knob rather than something to run with.
chad sets **no** `MLX_*` runtime variables, so there is nothing to opt out of:
`MLX_METAL_FAST_SYNCH`, `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` were each
measured end-to-end and every setting was *slower* than mlx's own defaults.
Export them yourself if you want to experiment; mlx reads them directly.

### Speculative decoding & kernel knobs

Everything in this block is **speed only**. Each one is bit-exact (or, for the sampled
paths, distribution-exact) with the path it replaces, so these are bisection and A/B
knobs — not something to run with. If output changes when you flip one, that is a bug.

```bash
CHAD_NO_MTP=1             uv run chad  # disable MTP self-speculative decoding
CHAD_MTP_ADAPTIVE=0       uv run chad  # fixed draft width instead of the adaptive schedule
CHAD_MTP_DRAFT=2          uv run chad  # force a draft width (implies adaptive off)
CHAD_MTP_MAX_DRAFT=6      uv run chad  # cap the adaptive schedule's width (max 8)
CHAD_MTP_H=…              uv run chad  # seed the depth policy's cost model
CHAD_NO_DRAFT_SHORTLIST=1 uv run chad  # full-vocab readout for the draft chain
CHAD_USE_PLD=1            uv run chad  # OPT-IN: wide prompt-lookup decoding
CHAD_NO_QSDPA_WIDE=1      uv run chad  # disable the S>1 tier of the fused attention kernel
CHAD_NO_QSDPA_WIDE_SGM=1  uv run chad  # disable just its split-head variant
CHAD_QSDPA_WIDE_SGM_RT=1  uv run chad  # force the RT-split wide kernel instead of the one-read form
CHAD_MTP_PATH=/path.safetensors uv run chad  # explicit MTP head sidecar (default: found beside the weights)
CHAD_NO_KERNEL_WARM=1     uv run chad  # skip warming verify-width attention kernels at load
```

- **`CHAD_NO_MTP`** — disables **MTP self-speculative decoding** (`mlx_mtp.py`), on by
  default wherever it can engage. chad drafts several tokens with the checkpoint's own
  trained multi-token-prediction head (loaded as a sidecar), verifies them in one batched
  forward, and accepts by exact rejection sampling — so greedy output is *token-identical*
  to the unspeculated path and sampled output keeps the model's true distribution at any
  temperature. It engages only on checkpoints that ship a head — the shipped model does;
  an arbitrary `--model` almost certainly does not. Measured on an M4 Pro at
  temp 1.0: **1.38× on quote-heavy spans, 1.11× on novel code, 1.0× on free prose** — it
  speeds up predictable text and costs nothing on unpredictable text.
- **`CHAD_MTP_ADAPTIVE` / `CHAD_MTP_DRAFT` / `CHAD_MTP_MAX_DRAFT` / `CHAD_MTP_H`** — the
  draft-depth schedule. By default a per-round cost model picks the depth from recent
  acceptance and, on a full-accept streak, jumps onto the measured flat verify plateau
  (an S≥10 verify costs about the same through S=32). Where acceptance is poor — temp-1
  thinking, cold content — it collapses to depth 1–2 or a free skip, so a bad regime
  degrades to fixed-width behavior rather than below it. `CHAD_MTP_ADAPTIVE=0` restores a
  fixed width; `CHAD_MTP_DRAFT=N` forces one and implies adaptive off (an explicit width
  is an order); the other two override the schedule's cap and its cost seed.
- **`CHAD_NO_DRAFT_SHORTLIST`** — the greedy draft chain reads its next token from a 2-bit
  shortlist rather than the full vocabulary, because that full-vocab `lm_head` read was
  ~70% of the head-step cost. Draft-side only: a shortlist miss costs a rejected draft
  token, never a wrong output token. This restores the full readout for bisection.
- **`CHAD_USE_PLD`** — turns **wide prompt-lookup decoding** back on. It was the default
  before 2.0.0 and is now opt-in, because PLD drafts from *context recurrence* and can
  therefore only accelerate text that already appeared. On real agentic traces that is a
  minority of what this agent generates: ~62–66% of generated tokens are `<think>`, and
  reasoning prose replays at only ~2.3%. Whole-session contribution measured at **+2.2%**
  of generated tokens. It does not compose with MTP — one generate loop each — so on a
  self-speculating checkpoint MTP is strictly the better of the two, and this flag is how
  you measure the other arm.
- **`CHAD_NO_QSDPA_WIDE` / `CHAD_NO_QSDPA_WIDE_SGM`** — disable the multi-token (S>1) tier
  of the fused quantized-KV attention kernel (`mlx_qsdpa.py`), which serves speculative
  verification and prefill. Without it those steps fall back to dequantizing the whole
  cache, which the wide path beats by 1.4–1.8× at 32k+. Bit-exact with that fallback.
  (`CHAD_QSDPA_WIDE_KERNEL=1` forces the single-kernel variant instead of the split-head
  one — a kernel-selection knob for measurement.) The bigger hammers are still
  `CHAD_NO_QSDPA`, in [Safety & A/B opt-outs](#safety--ab-opt-outs), which disables the
  fused kernel entirely.
- **`CHAD_QSDPA_WIDE_SGM_RT`** — the wide tier has two forms. The default is *one-read*
  (K/V bytes touched once, all row tiles resident); this forces the *RT-split* form (one
  8-row tile per threadgroup, K/V read once per tile), which is what runs anyway at S>4.
  Both are exact; the knob exists to A/B them at a given width.
- **`CHAD_MTP_PATH`** — an explicit path to the MTP head's `.safetensors` sidecar. Unset,
  the loader looks beside the weights (the shipped model bundles it as `mtp.safetensors`).
  Useful when building a head yourself with `python -m chad.mlx_mtp`.
- **`CHAD_NO_KERNEL_WARM`** — the attention kernel is templated on its verify width, so a
  width that has never run means a Metal compile lands on the critical path of a real
  step. Load warms exactly the widths *this* configuration can produce (MTP verifies at
  one width; wide-PLD at any of 9–32), rather than the union of everything. Opting out
  moves those compiles into your first few steps.

### Dev & instrumentation

Not supported surface — they exist for debugging chad itself, and their formats can change
between releases.

```bash
CHAD_TRAJECTORY_JSON=/tmp/traj.json uv run chad  # record an ATIF trajectory (pure observer)
CHAD_SPILL_DIR=/tmp/spill           uv run chad  # where large tool outputs spill to disk
CHAD_DUMP_RENDER=/tmp/prompt.txt    uv run chad  # dump the fully-rendered prompt each step
CHAD_PREFILL_TRACE=/tmp/pf.jsonl    uv run chad  # per-step prefill/cache telemetry
CHAD_CHECKPOINT_DIR=/tmp/ckpt       uv run chad  # relocate the shadow-git edit checkpoints
```

- **`CHAD_CHECKPOINT_DIR`** — where the shadow-git repositories backing `/undo` and
  `/restore` live (default `~/.chad/checkpoints`, keyed per workspace). It exists so a test
  or eval suite never writes real home state. Note this is *not* `~/.chad/history`, which is
  the TUI's prompt-history file.

### Tree-sitter tags (ambient structure)

The `bash_read_skeleton` lever's one-line symbol maps and definition pointers come from
a tree-sitter tags index. Defaults are tuned for a big repo on a memory-tight machine;
you rarely need to touch this.

```bash
CHAD_REPOMAP_WORKERS=4   uv run chad  # subprocess workers for a cold repo scan (1 = serial)
```

- **`CHAD_REPOMAP_WORKERS`** — how many `python -c` subprocess workers a cold whole-repo
  tag scan shards across (`repomap.py`; default: cores−2, capped at 8). Workers import
  only `chad.repomap` — never the MLX engine. Tags persist per repo under
  `~/.chad/cache/repomap/` (mtime-validated per file), so warm sessions skip the scan.

### Voice mode (`/speech`)

Voice mode is all on-device: [Parakeet-on-MLX](https://github.com/ml-explore/mlx) (vendored)
transcribes your mic, macOS `say` speaks the replies. It needs the `speech` extra — see
[Installing](../README.md#installing--upgrading) — and nothing leaves the machine.

```bash
CHAD_VOICE="Daniel"        uv run chad  # macOS `say` voice (default: the system voice)
CHAD_SPEECH_RATE=200       uv run chad  # `say` rate in words/minute (default: the system rate)
CHAD_STT_QUANT=4           uv run chad  # ASR weight quantization: 8 (default), 4, or none
CHAD_STT_MODEL=<hf-repo>   uv run chad  # override the ASR checkpoint
CHAD_SPEECH_WORDS=/path.json uv run chad # personal word table (default ~/.chad/speech_words.json)
```

- **`CHAD_VOICE`** — an installed macOS voice name. An unknown name is refused at startup
  with a did-you-mean rather than silently falling back; `say -v '?'` lists what you have.
- **`CHAD_STT_QUANT`** — `8` by default. `4` halves the ASR weights again but is **opt-in**:
  clean-audio testing can't rule out degradation on a noisy mic. `none` keeps full precision.
- **`CHAD_STT_MODEL`** — defaults to `mlx-community/parakeet-tdt-0.6b-v3`. Anything else is
  unsupported; the vendored decoder is written to this model's output contract.
- **`CHAD_SPEECH_WORDS`** — the JSON word table that teaches the transcriber your jargon
  (project names, library names). Relocating it is mostly useful for keeping a benchmark
  from inheriting your personal vocabulary.

### Sessions

Every conversation is persisted per *working directory*, so `-c` in a project resumes that
project's thread and nothing else:

```
~/.chad/sessions/<cwdhash>/<session_id>.json   one file per session
~/.chad/sessions/<cwdhash>/index.json          title / last-updated / turn count
```

The newest **20** sessions per directory are kept; older ones are pruned on save. Both
files are created mode `0600` — they hold full tool arguments and results (see [Session
log & privacy](#session-log--privacy)).

```bash
uv run chad -c            # resume this directory's most recent session
uv run chad --resume      # list recent sessions, pick one by number (needs a TTY)
```

**Resuming forks; it never overwrites.** Both flags seed a *fresh* conversation with the
old messages, and that conversation mints its own `session_id`. The session you resumed
from is left exactly as it was, so branching off an old thread can't destroy it and you
can resume the same starting point twice. The practical consequence: a long session you
resume repeatedly leaves several sessions behind, which is what `--resume`'s numbered
list is for.

### Session log & privacy

Diagnostics log: when enabled, each session appends throughput numbers and a readable
trace — the user query, tool-call args (including bash commands and write/edit content),
and result previews — to `~/.chad/session.log`. It's size-bounded (rotated, 5 MB × 3)
and passes previews through a best-effort secret redactor, but it still records
command/file previews in plaintext outside the repo, so treat it as sensitive.

**Privacy-first default: the trace is OFF.** chad is a local, single-user agent, so
nothing leaves your machine — but because the log lands plaintext previews under
`~/.chad`, it is opt-in. Set **`CHAD_SESSION_LOG=1`** (any truthy value) to turn it on;
the same flag also enables the persistent input history at `~/.chad/history` (mode
`0600`). When it's off, chad installs a null handler and won't create `~/.chad` for the
log's or history's sake.

**`CHAD_NO_SESSION_LOG=1`** remains a hard kill switch: if set it forces both the log
and the history off, and wins even when `CHAD_SESSION_LOG` is also set. (For the same
privacy reason, the resumable conversation store under `~/.chad/sessions/` — which holds
full tool args and results, and is written only when you use `-c`/`--resume` — is created
mode `0600`.)
