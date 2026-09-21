# Configuration & reference

*Steering chad (project instructions, Agent Skills, MCP servers, plan mode) and the full
flag/env-var reference. For the basics, see the [README](../README.md).*

## Project instructions (CLAUDE.md / AGENTS.md)

Standing instructions for a project — conventions, the commands you want used, things not
to touch — go in a markdown file at the root of that project. chad looks for exactly two
names in the **working directory**, in this order:

| Order | File        |
| ----- | ----------- |
| 1     | `CLAUDE.md` |
| 2     | `AGENTS.md` |

**The first one that exists wins, and that is the only one read.** `CLAUDE.md` shadows
`AGENTS.md`; they are never merged, and a repo that carries both is only using the first.
Only the working directory is searched — not parent directories, not `~` — so the file
you get is the one belonging to the project you launched chad in.

The first **4000 characters** are used and the rest is dropped silently, so put what
matters at the top. The text is appended to the system prompt under a
`# Project instructions (<filename>)` heading, below the cache boundary with the other
per-project context (working directory, workspace listing). That placement is the whole
cost story: the static half of the prompt stays byte-identical across projects and keeps
its global checkpoint, while your instructions are part of the few-hundred-token project
tail that is prefilled once and then restored from that project's own warm-start
checkpoint — not re-sent on every turn.

**`/init` writes one for you.** It orients itself with `bash`, reads whichever of
`README` / `pyproject.toml` / `package.json` / `go.mod` / `Cargo.toml` / `Makefile` exist,
and writes a concise `CLAUDE.md` — an overview, the main components, the *actual*
build/run/test commands copied out of the config it read, and any conventions worth
noting. If a `CLAUDE.md` is already there it reads and improves it rather than clobbering
it. The file is an ordinary write, so it goes through the usual confirmation in `normal`
mode.

## Agent Skills (agentskills.io)

chad implements the open [Agent Skills](https://agentskills.io) format, so a skill
authored for Claude Code (or any compatible client) works here unchanged. A *skill* is a
folder with a `SKILL.md`: YAML frontmatter (`name` + `description`, plus optional
`license`/`compatibility`/`metadata`/`allowed-tools`) followed by markdown instructions,
optionally bundling `scripts/`, `references/`, and `assets/`.

**Where chad looks** (project skills override user skills on a name clash):

| Scope   | Paths |
| ------- | ----- |
| Project | `./.agents/skills/`, `./.claude/skills/` (relative to the working dir) |
| User    | `~/.agents/skills/`, `~/.claude/skills/` |

**You choose the skill, not the model.** Every installed skill is a slash command:

1. Dispatch: type `/` and the completion menu lists every skill by name and
   description, alongside chad's builtins. `/skills` prints the same list with any
   discovery warnings. None of this reaches the model, so it costs nothing in context.
2. Load: `/ship` submits that one skill's instructions as your turn, wrapped in
   `<skill>` with its directory and bundled files listed. Anything you type after the
   command is the task: `/investigate the flaky truncation test`.
3. Resources: referenced `scripts/`/`references/`/`assets/` files are read on demand
   against the skill's directory, from `bash` (`sed -n '1,120p' <file>`).

This is a deliberate divergence from the spec's tier-1 disclosure, where a catalog of
every skill's description rides in the system prompt for the model to select from. That
catalog measured **4,751 tokens against 62 installed skills, 60% of chad's entire system
prompt**, paid on every turn of every session, so a small model could guess at a choice
you can make instantly from a menu. Dropping it took the system prompt from 7,975 tokens
to 2,824. On a 24 GB box whose usable window is ~50k, that is 10% of the window back.

The cost you *do* still pay is the skill body itself, and it can be large, a big
Claude Code skill runs past 40k tokens, most of the window. chad prints the exact token
count when you load one (`loaded skill ship (41,238 tokens)`) so the bill arrives when
you incur it, rather than as an unexplained compaction three turns later.

Parsing is lenient (a name that doesn't match its directory, an over-long field, or an
unquoted `colon: value` in YAML loads anyway, with a warning); only a missing description
or unparseable YAML is skipped. A loaded skill's turn is exempt from context compaction,
you asked for that guidance by name, and nothing would reload it, and re-running the same
command notes that it is already loaded instead of sending a second copy. Implementation:
`src/chad/skills.py` (discovery/parse/dispatch/load), with the menu in `tui.py` and the
compaction guard in `compaction.py`.

## MCP servers (modelcontextprotocol.io)

Agent Skills add *instructions*; **MCP** adds *tools*. chad can connect to external
[Model Context Protocol](https://modelcontextprotocol.io) servers (a GitHub server, a
Postgres server, a company's internal API server, or hosted connectors like Linear and
Slack) and expose their tools to the model alongside its builtins. chad uses the official
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
servers are deferred until you log in (see below), and, most importantly, **project-scope
`./.mcp.json` servers do not start until you `/mcp trust` the project** (next section).
User-scope `~/.chad/mcp.json` servers are authored by you and auto-connect.

**Project trust.** Dropping a `./.mcp.json` into a repo does **nothing** until you run
`/mcp trust` in that directory. This is deliberate: a project file is content you may have
just cloned, and a stdio server is an arbitrary local command; an untrusted repo must not
be able to auto-launch a subprocess the moment you open chad in it. Until trusted, its
servers show gated in `/mcp` (`project server not started — project not trusted (run /mcp
trust)`) and contribute no tools. `/mcp trust` records the project's absolute path in
`~/.chad/trusted_mcp.json` (mode `0600`); the path is the trust anchor, so moving the repo
to a new directory re-prompts. User-scope servers are exempt (you wrote them).

> `CHAD_MCP_FULL_ENV`, a stdio MCP subprocess inherits only a **minimal env allowlist**
> by default (`PATH`, `HOME`, `LANG`/locale, `TMPDIR`, `SHELL`, `USER`, …, enough to find
> its binary and start), **not** chad's full environment. A user-configured server runs an
> arbitrary local command and has no business inheriting your API keys, cloud tokens, or
> provider creds. A server that genuinely needs one var declares it in its config `env:`
> block (merged in, and it wins). Setting `CHAD_MCP_FULL_ENV=1` restores the full
> parent-environment inherit for the rare server that needs it, but it hands **every**
> secret in chad's environment to **every** stdio server, so leave it unset unless you know
> exactly which server needs what.

> **Auth:** static bearer/PAT tokens via `headers` work out of the box. Hosted connectors
> that require **OAuth** (Linear, Slack, Atlassian, most SaaS) are supported behind an
> opt-in flag; see below.

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
to approve), OAuth servers do **not** auto-connect; that would hang an agent turn or an
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
  stdio/bearer/HTTP paths are unchanged; none of the OAuth code runs.
- Headless / no-browser sessions are **never blocked**: an OAuth server that can't complete
  an interactive login simply contributes no tools.
- Token values are never logged. The token file is created `0600` from the first write.

**How they behave in the harness:**

- Namespaced `mcp__<server>__<tool>`, so server tools can't collide with chad's
  builtins (`bash`/`edit`/…) or with each other.
- Same validation path as builtins: each tool's `inputSchema` drives the typed-coerce
  + self-repair loop (`"3"`→`3`, missing-required detection), no schema duplication.
- Confirmation gate. A tool the server marks `readOnlyHint` runs without a prompt;
  every other MCP tool is treated as mutating (the safe default: it might write files, hit
  an API, or send a message), so it asks first and shows its arguments. In `--plan` mode
  only read-only MCP tools run.
- Graceful degradation. A server that's missing, misconfigured, slow to connect, or
  that crashes mid-session contributes no tools and never takes the agent down. `/mcp` shows
  each server's transport (stdio/http), status, tools (with read-only markers), and warnings.

Implementation: `src/chad/mcp.py` drives the official `mcp` SDK behind a synchronous,
cwd-keyed registry: the SDK's async event loop runs in a background thread (one
`anyio` BlockingPortal) and every call is marshalled onto it, with the per-call timeout
inside the coroutine so a hung server can never wedge the agent. Wired into
`tools.active_schemas`/`dispatch_for`/`is_mutating`, the validator (`validate.py`), and the
agent loop (`agent.py`).

## Plan mode

`--plan`, or shift-tab round to `plan mode`, makes the session read-only with exactly one
exception: `write` and `edit` are allowed when the path resolves inside **`./plans/`**.
Every other mutating tool — `bash` included, so no commands run — is refused with a note
telling the model to investigate read-only and write its plan instead. A write under
`plans/` is the expected move, so it does not ask for confirmation; nothing else in plan
mode gets the chance to. The gate resolves symlinks against the real working directory, so
a `plans` entry that is itself a link cannot carry a write somewhere else.

**What you get back is a file.** A change request is answered with one self-contained
`plans/NNN-kebab-title.md` (continuing whatever number sequence is already there), holding
everything an executor needs without the chat: context, file paths with current-state
excerpts, numbered steps, verify commands, and what is out of scope. `./plans/` is created
on that first write, and it is an ordinary directory in your repo — commit it or add it to
`.gitignore`, chad does not care. A plain *question* asked in plan mode is answered in
prose instead; it does not manufacture a plan file.

**Handing the plan back.** When a plan-mode turn finishes having written a file, chad
prints `plan ready → <path>` and waits. Type to steer (the plan turn continues), or press
**ctrl-g** — or run **`/accept`** — to accept it: the context is cleared, the session drops
back to the permission mode it had before plan mode, and a fresh turn starts with an
instruction to read that file and execute each step, running the verification commands at
the end. Accepting is the only handoff; a plan left un-accepted is just a file on disk.

## Slash commands

Typed in the TUI (`/` opens a completion menu listing these alongside every installed
skill). Most are local to the harness and cost nothing in context; `/init` and `/accept`
are the two that start a real turn.

| Command | What it does |
| ------- | ------------ |
| `/help` | commands & keybindings |
| `/init` | analyze the project, write `CLAUDE.md` (a real turn — the model does the work) |
| `/skills` | list installed Agent Skills (run one with `/<name>`) |
| `/mcp` | MCP server status |
| `/mcp trust` | trust this project's `.mcp.json` servers |
| `/mcp login` | authenticate an MCP server (OAuth) |
| `/compact` | reclaim context now |
| `/ctx` | where the context window is going, in tokens |
| `/undo` | revert files to the last edit checkpoint |
| `/restore` | list edit checkpoints; `/restore <hash>` reverts to one |
| `/resume` | list recent sessions; `/resume <n>` forks one |
| `/reset` | clear the conversation + KV cache |
| `/clear` | clear the conversation + KV cache |
| `/model` | show model + context window |
| `/mode` | cycle permission mode |
| `/speech` | toggle voice mode — all-local STT (Parakeet-on-MLX) + TTS (`say`) |
| `/accept` | accept a pending plan and implement it |
| `/exit` | quit chad |
| `/quit` | quit chad |

A builtin always wins a name clash with a skill, so `/<name>` reaches a skill only when no
builtin owns that name.

## Context window (agentic coding needs room)

By default the harness uses the model's **full native window** instead of an arbitrary cap,
then lets the RAM-aware governor decide how much of it this machine can actually spend. On
the shipped model native is **262k**, so there is nothing to extend; `CHAD_MAX_CONTEXT` is
mostly a knob for an arbitrary `--model`, where it requests a larger window and
**YaRN-extends** the checkpoint past native if its config supports it, capped at the model's
documented max. Setting it *below* native is the one use that bites on the shipped model:
it lowers the ceiling the governor sizes against. KV cache grows lazily, so a large window
costs nothing until tokens fill it, and the cache is **quantized to 8-bit by default** (half
the fp16 footprint; `CHAD_KV_BITS=0` restores fp16).

How much it costs is a property of the model's attention design, and the shipped one is
unusually cheap here. Qwen3.8-27B is a **hybrid SSM/attention** model: 48 of its 64 layers
are GatedDeltaNet and carry a *fixed-size* recurrent state no matter how long the context
gets, so only the 16 full-attention layers grow with it. Measured footprint, at the 8-bit
default:

| Context | KV cache (8-bit, default) | Notes |
|---|---|---|
| 32k | 1.1 GB | |
| ~56k | 2.0 GB | roughly where a 24 GB Mac's window lands on the 3-bit `--model` alternative |
| 128k | 4.6 GB | |
| ~150k | 5.2 GB | roughly where a 24 GB Mac's window lands on the shipped ternary weights |
| 262k (native) | 9.1 GB | the checkpoint's max; unreachable on 24 GB |

That is **34,816 bytes per token**. A pure-attention transformer of the same shape (all 64
layers keeping per-token K/V rather than 16) would spend ~131,000 bytes per token at the
same quantization, about **4×** as much, and its 128k row alone would exceed the whole
machine. Trading trimmability for a flat memory profile is what buys the window
([design](design.md#the-cache-only-appends)); `CHAD_KV_BITS=0`
roughly doubles the numbers above. When the prompt nears the window, old verbose tool outputs
are compacted.

8-bit KV used to cost ~20-30% throughput, because mlx_lm's quantized attention is unfused,
which is why it was opt-in through most of 1.x. chad now ships a fused quantized-KV decode
kernel, installed automatically when the model's attention shape is covered (the shipped
model's is), which makes the quantized cache *faster* than fp16 at long context on top of
halving the RAM. That is why it is the default. The crossover was measured on the shape that
was shipped at the time (the 35B, at 32k: 60.2 vs 55.8 tok/s), and the mechanism is the
kernel, not the weights, so it holds on any covered model; `chad-bench` reports what your
machine is actually getting. `CHAD_KV_BITS=0` restores the fp16 cache; `CHAD_NO_QSDPA=1`
keeps the quantized cache but disables the fused kernel (debug only; that combination is
the old slow path).

```bash
CHAD_MAX_CONTEXT=131072 uv run chad   # cap the window at 128k (the shipped model is 262k native)
```

## Advanced (env vars)

The rarely-touched tuning knobs live in environment variables so they stay off the main
`--help`. Same capability, sane defaults when unset:

```bash
CHAD_MAX_CONTEXT=131072 uv run chad      # request a window (YaRN-extends a smaller --model past native)
CHAD_KV_BITS=0          uv run chad      # fp16 KV cache (8-bit fused is the default where covered)
CHAD_KV_CACHE_MAX_GB=4  uv run chad      # disk budget for the warm-prefix KV checkpoints (default 8; 0 = unlimited)
CHAD_CTX_LIMIT=28000    uv run chad      # force the compaction threshold (overrides the RAM-aware default)
CHAD_CTX_SAFETY=0.95    uv run chad      # the single headroom lever: fraction of the Metal budget
                                         # the auto-sizing may spend (default 0.975 — hold back 2.5%)
CHAD_CTX_SLOPE_FACTOR=1.5 uv run chad    # A/B knob: per-token cost multiplier for that auto-sizing
                                         # (default 1.0 — a token's marginal cost is its KV cost)
CHAD_MODEL=/path/to/mlx-model uv run chad  # power-user escape hatch: run a different MLX model
                                         # (also: --model auto|<repo> — the CLI twin, wins over this)
CHAD_PREFILL_CHUNK=1024 uv run chad      # force a fixed prefill chunk (default: adaptive — 512 on the
                                         # shipped dense model, decaying to 256 as context+pressure grow)
CHAD_NO_MEMORY_CLAMP=1  uv run chad      # A/B knob: skip the Metal allocator clamps installed at load
```

**`CHAD_KV_CACHE_MAX_GB`** is the only one of these that spends *disk* rather than RAM.
The warm-prefix checkpoints ([benchmarks](benchmarks.md#the-second-session-in-a-project-starts-warm))
live in `~/.cache/chad/kv`, one file per distinct system prompt, so a machine with many
projects accumulates them. chad LRU-evicts that directory down to this budget (default
**8 GB**) after each write, always protecting the file it just wrote and the live warm
prefix, so trimming the budget costs a cold first turn in the least-recently-used projects
and nothing else. `0` disables eviction. Each checkpoint has a fixed ~51 MB floor from the
serialized recurrent SSM state, which is why the budget is in GB and not MB.

By default the auto-compaction threshold (when chad reclaims old context, a full
re-prefill on this non-trimmable cache, so we do it as rarely as RAM allows) is **sized
automatically** from the live Metal memory budget and the model's measured per-token
cost, then capped at the model's window. Three things are subtracted before the division:
the resident weights, a headroom band (`CHAD_CTX_SAFETY`), and the prefill transient, the attention scratch that is live at the same moment the cache is. That last one is fixed,
not per-token: it climbs with context and then flattens once the adaptive chunker starts
shrinking the chunk (measured: 1.8 GB at 8k, 4.15 GB at 49k, flat thereafter), and past
that point peak memory grows at exactly the KV rate. On a 24 GB Mac the shipped ternary
weights (7.2 GB) plus the drafter (1.1 GB) and that 4.3 GB transient spend ~72% of the
budget before the first cached token, which is why it lands around ~150k rather than the
262k native window; the 3-bit alternative's 12.3 GB of weights spent 87% and landed near
~56k. It
self-calibrates per machine: less RAM compacts sooner, more RAM runs nearer the full
window. `CHAD_CTX_LIMIT` forces an
exact threshold (used by tests); `CHAD_CTX_SAFETY` (default 0.975) is the single
headroom lever, the fraction of the Metal budget the sizing may spend, so lower it if
you run other memory-hungry apps alongside chad; `CHAD_CTX_SLOPE_FACTOR` tunes the
per-token multiplier (1.0 recovers the raw-KV sizing).

The banner states the window you will **actually get**, not the checkpoint's native one.
On a memory-tight box the two differ by more than 2x (a 262k model on a 24 GB Mac gets
tens of thousands of tokens once the weights and the KV cache are paid for), and the
native number is context the run can never spend. When the governor is what bound it,
the banner says so: `84k of 262k context`.

`--model` (or `CHAD_MODEL`) takes `auto` (the shipped model) or any Hugging Face repo id /
local MLX model directory, and nothing else. There are no size shorthands: `--model 9b` is a
literal (and nonexistent) repo id, not an alias, and fails as one. Pointing it at other
weights is supported and kept on purpose: one shipped model is a default, not a restriction,
but the harness is *tuned* to the shipped model, so the realistic cost is throughput, not
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

chad ships exactly one: [`nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX`][model],
Qwen3.8-27B (a dense `qwen3_5` hybrid: 64 layers, 48 GatedDeltaNet and 16 full attention)
in Prism ML's ternary build. Every projection is Hadamard-rotated offline and stored as
2-bit affine group-128 whose three levels are {−s, 0, +s}: **7.2 GB resident**, 262k native
context. The repo also carries the DFlash2 block drafter, pre-quantized to 4-bit in
`dflash/` (~1.1 GB resident), so one download gets the model and its
[speculative decoder](#speculative-decoding--kernel-knobs) with nothing built on first run.

[model]: https://huggingface.co/nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX
[q3]: https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX

The rotation is folded into the weights, so the matching transform has to hit the
activations at runtime and be inverted after the embedding lookup. mlx-lm's plain affine
loader finds tensors of exactly the right shapes, skips it, and returns plausible garbage
without raising; chad routes the pack on its declared `model_type` to its own loader
(`prism_pack.py`), which does not import the Python the upstream pack bundles. The repo
is chad's repack of [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
(Apache-2.0; created using Bonsai by Prism ML): text-only (the pack's vision tower
dropped), the base model's tokenizer in place of the pack's (same vocabulary, different
merges: the pack's trips transformers' regex warning and diverges on combining marks),
the chat template's `reasoning_effort` defaulting to medium rather than xhigh, and the
drafter bundled. Pointing `--model` at the upstream pack directly also works: the loader,
the fast-path and the verify kernel attach the same way, the drafter is borrowed from the
shipped repo (`mlx_dflash.DONORS`), and the engine passes medium unless
`CHAD_REASONING_EFFORT` says otherwise, since think-token decode is two thirds of wall on
this model.

**chad targets 24 GB Apple Silicon and nothing smaller.** There is no low-RAM fallback,
because there is no second model to fall back to. On a smaller box chad prints a one-line
warning at startup and runs anyway (it advises, it does not gate), but ~8 GB of weights and
drafter plus the ~4.3 GB prefill transient leave little for a KV cache, so the window
shrinks toward its floor. At 24 GB the honest figure is roughly ~150k of the model's 262k
window; the banner states what you actually got, and the
[context window](#context-window-agentic-coding-needs-room) section explains the sizing.

**The 3-bit alternative.** [`nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX`][q3] is the
same model at 3-bit group-64 with `lm_head` held at 5-bit (the bit map follows what the
calibrated GGUF builds of this checkpoint agree on: the head is a second full 1.27B-param
tensor worth protecting, the embedding a lookup table that is cheapest to cut), ~12.3 GB
resident, the same drafter bundled, and ~56k of window on 24 GB. It is one flag away
(`--model`), and it is the quality reference: on code, teacher-forced perplexity is 3.99
against the ternary's 4.49 (+12%), while the private eval tiers tie at 56/56. Every projection's
weights are on the critical path for every token on a dense model, so the governor's
34,816 B/token prices a gigabyte of weights at about **29k tokens of context**; the 5 GB the
ternary build gives back is where its window comes from, and its speed matches because the
verify kernel and the drafter carry over.

To run different weights through the same engine:

```bash
uv run chad --model mlx-community/Some-Other-Model-4bit   # any HF repo id
uv run chad --model /path/to/local/mlx-model              # or a local dir
CHAD_MODEL=/path/to/local/mlx-model uv run chad           # env equivalent
```

Precedence is `--model` → `CHAD_MODEL` → the shipped default. Both are honored without
argument. Expect to lose speed rather than correctness: the tuning that is fitted to the
shipped checkpoint (the DFlash2 drafter, the fused-attention coverage, the fastpath's architecture
check, the measured per-token KV cost the governor sizes against) either declines to
install or falls back to a stock path. The harness itself does not change.

#### What the engine does with the ternary weights

- the decode fast-path fuses `gate|up`, `qkv|z` and `q|k|v` behind **one** rotation each,
  and folds the sign vectors into the weights once, at install: the residual-width vectors
  into the two layernorms, `down_proj`'s into the rows of `up_proj` (a sign flip of those
  rows' scales and biases). All exact, since the vectors are ±1. A rotation is then one
  kernel, the transform itself in the activation dtype, on **every** path: the serial
  step, the width-2..8 verify forward a drafted round actually spends its time in, and
  prefill. The pack's own rotation is four (two fp32 casts, the sign multiply, the
  transform). Serial and verify forwards also rotate identically now, so they write the
  cache at the same precision. `CHAD_PRISM_ROT_FP32=1` restores the fp32 transform
  everywhere (an A/B arm: teacher-forced code NLL moves by ≤0.002 in no consistent
  direction, and the native rotation prefills 1-3% faster);
- the small-M verify kernel (`mlx_qmm_mma.py`) covers 2-bit g128, which is what makes an
  8-wide DFlash2 verify cost ~2.2 serial steps here as on the 3-bit. mlx's stock 2-bit
  matmul re-pays the weight read per row, made that verify 9× a decode step, and turned
  drafting into a net loss (12 tok/s against 18 serial) until the kernel covered it. Past
  one 8-row tile the same cliff came back (a 9-token forward cost 5.4 steps against 2.2
  at 8), so widths 9-24 now run as several 8-row kernel calls: 1.2-1.8× faster on exactly
  the forwards an agent step is made of (a warm tail, a short tool-result suffix);
- the speculative schedule is seeded with a round-cost ladder measured on *this* weight
  width, and each turn starts from the ladder the last one measured. The 3-bit ladder
  prices every width past one 9-18% too cheap here. On low-acceptance prose (20-30% of
  drafted positions accepted) the schedule sits at break-even with serial decoding on
  either seed, within run-to-run noise, so this is a correctness-of-the-model fix, not
  a measured tok/s one;
- when the fast-path declines a Prism pack it says so, and why, as a warning. An unfused
  pack decodes correctly and several times slower, which is not a thing to learn from a
  stopwatch;
- the residual stream runs in bf16. As the upstream pack ships, its fp32 norms and fp16
  scales promote every activation to fp32 from the first RMSNorm on; the loader casts the
  side tensors once, as mlx-lm's own loader does, at no cost in NLL (1.502 vs 1.500);
- the serial step is bounded by mlx's 2-bit GEMV rate rather than by dispatch, so the
  fast-path's kernel-count win is smaller here than on the 3-bit (21 vs 18 tok/s); the
  weights, the verify and the window are where the build pays.

### Smoke test (`chad prove`)

```bash
uv run chad prove
```

Downloads the shipped model if it isn't cached, then drives four tiny fix-it tasks
end-to-end against it: a real agent loop, real edits in a scratch directory, each verified
by a check script re-written from read-only sources so an agent edit can't spoof a pass,
and prints what worked, with time-to-first-token, decode speed and wall-clock per task.
`results.json` lands in the invoking directory.

Two deliberate rigidities. It **pins the shipped model**, ignoring `--model` and
`CHAD_MODEL` (it says so when you set one): the question is "does the thing I am about to
run work on this machine", and a smaller stand-in cannot answer it. And once the weights are
present it **goes offline** (`HF_HUB_OFFLINE` plus an in-process socket guard), so a task
cannot quietly reach the network. Exit codes: `0` all passed, `1` a task failed, `2` the
preflight stopped.

These are smoke tests, not a benchmark: on working hardware they should essentially never
fail, so a failure points at the install or the machine rather than at the model's ability.
Reach for it first when something is broadly wrong, before reading
[Troubleshooting](troubleshooting.md).

### Alternate backend (remote)

The default backend is the in-process MLX engine, the whole point of chad (a persistent
prefix KV cache on-device). `--backend llama` instead drives the *same* harness against a
remote **llama.cpp** server's raw `/completion` endpoint (token-id prompts). This is the
arm used when chad runs inside a Linux benchmark container against a GGUF served on a GPU
box, where MLX can't run. It's lossy relative to the in-process engine (the KV cache lives
in the server, so warm-prefix checkpoints are no-ops) but keeps real
cache telemetry and passes `<think>` back verbatim, not a general "use a cloud model" path.

```bash
uv run chad --backend llama --base-url http://<host>:8081   # or CHAD_LLAMA_BASE_URL
```

- `--base-url` / `CHAD_LLAMA_BASE_URL`: the llama-server origin (bare, no `/v1`);
  required for `--backend llama` (the flag wins; the env var is the fallback).
- `--tokenizer` / `CHAD_TOKENIZER`: HF repo whose tokenizer matches the served
  model's vocab; required for a GGUF server (GGUF repos ship no tokenizer).
- `--api-key-env NAME`: the *name* of the env var holding the API key (read from that
  var, never passed on the command line). Omit for a local endpoint that needs no key.

### Sampling & reasoning effort

chad decodes **greedily by default** (temperature 0): reproducible, and the cheapest path
through the engine. That default has one field-measured failure mode worth knowing: a
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

- `CHAD_MIN_P` / `CHAD_TOP_P` / `CHAD_TOP_K`: anti-confabulation knobs for a heavily
  quantized model, all off by default. They trim the sub-noise-floor logit tail without
  touching temperature, which is usually what you want when a small quant invents an API.
- `CHAD_PRESENCE_PENALTY`: a flat score penalty on every already-emitted token.
  Ships at **0.0** and is worth leaving there, even though model cards suggest up to 1.5:
  those ranges are written for chat, and code is inherently repetitive: identifiers,
  keywords and punctuation *must* be reused. Measured on one task at 1.5, the model spent
  45 steps in pure exploration, landed zero edits, and emitted visibly corrupted tool
  arguments. Treat it as a knob to probe, not a default to ship.
- `CHAD_REASONING_EFFORT`: `xhigh` | `medium` | `low`, passed to the chat template as
  a reasoning budget on checkpoints whose template accepts one (Qwen3.8). Unset, the
  argument is not passed *at all*, so a template without the knob is unaffected. This is a
  template-level request to the model, distinct from the harness-level
  [think-cap](#turn-budgets--think-cap) below, which force-closes a `<think>` run the model
  has already started.

All five sampler settings are applied as one call, so every path that builds an engine
(interactive and one-shot) honors the same set.

### Turn budgets & think-cap

A runaway-turn **governor** ends a turn that burns a lot of prefill without landing and
verifying a change; it nudges at ~50% of budget and, at ~80%, banks a one-line progress
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

- `CHAD_THINK_BUDGET`: soft-caps each step's `<think>` run at N tokens, force-closes it,
  and continues (escalates when the model is stuck); off by default.

  This is the adaptive think-cap, a smarter reasoning lever than the blunt `--no-think`:

  - `--no-think` kills **all** `<think>` for the whole session. It's the biggest,
    bluntest time saver (see [benchmarks](benchmarks.md), "the most effective time-to-done
    lever on well-scoped agentic work"), but it costs pass-rate on hard tasks that genuinely
    need to reason. All-or-nothing.
  - the adaptive think-cap (`--think-budget N` / `CHAD_THINK_BUDGET=N`) leaves reasoning
    **on**, and only trims a step whose `<think>` run runs past N tokens, force-closing it
    (prefix-safe, so the next step is a cheap append, not a re-prefill) so reasoning can't
    balloon. The cap **escalates** with the turn's stuck-signals: a genuinely hard step that
    keeps getting capped is given more room instead of being chopped repeatedly. So it keeps
    full reasoning on well-scoped work and only trims the rambling. Off by default today, flipping it on by default is an eval-gated decision.

  When the cap fires during a turn, the TUI status line shows a small **`✂N`** counter (N =
  steps trimmed this turn) alongside the live ↑prefill / ↓generated readouts, so you can see
  it acting. With the cap off (the default) nothing renders.
- `CHAD_THINK_CEILING`: the **close-and-continue** ceiling, and the one to reach for
  before `CHAD_THINK_BUDGET`. Where the think-cap force-closes `<think>` and *ends* the
  step (so the model re-derives its reasoning next step), this force-closes the runaway
  block and keeps decoding the action in the same step, the reasoning so far stays
  in context and nothing is re-derived. **Off by default**: force-closing `</think>`
  mid-generation is the most invasive thing the harness can do to the token stream, and
  the measured record says the bare loop doesn't need it. Steps it fires on are counted
  as *salvaged* in the session log.
- `CHAD_TURN_BUDGET_TOKENS`: the governor's cumulative-prefill budget per turn; defaults
  to 3× the context limit. Disable the governor entirely with `CHAD_NO_GOVERNOR=1` (below).
- `CHAD_TURN_BUDGET_S`: a wall-clock (seconds) variant of the same governor; off by
  default.
- `CHAD_MAX_GEN_TOKENS`: the hard ceiling on a single *step's* generation, 32768 by
  default. It is a backstop against non-repetitive runaway garble, not a reasoning lever:
  a literal decode loop is the repeat guard's job. The ceiling is deliberately high because
  a low one has a real loss class: a long chain of thought pins the cap while still inside
  `<think>`, and the step ends as discarded reasoning with no action.

> **`CHAD_PREFILL_TRACE=path.jsonl`** is a dev/instrumentation knob, **not** supported
> config: it captures one JSON row per engine prefill to the given path for measurement
> spikes. Each row also carries the loop overhead outside the engine: `render_s`
> (chat-template re-tokenization), `compact_s`, and `prev_tools` (the prior step's tool
> executions as `[name, seconds]` pairs), so a slow step can be attributed to prefill,
> tokenization, compaction, or a tool without guessing. Leave it unset in normal use.

### Harness levers

chad 2.0.0 ships **ten** result-channel levers, all ON by default, the survivors
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

| Lever | What it adds to the result channel |
|---|---|
| `env_manifest` | session-start toolchain inventory in the system prompt tail: which compilers/interpreters/package managers are present, with versions, and which commonly-probed ones are absent |
| `bash_read_skeleton` | a one-line symbol map the first time a source file's content comes back through `cat`/`sed`/`head`, and a definition pointer when a grep for a known symbol comes back empty |
| `bash_empty_diagnose` | why a command produced nothing: a `sed` range past EOF, or the pipeline stage whose filter matched |
| `bash_trim_keep_failures` | when long output is head/tail-trimmed, the failure rows from the omitted middle are kept verbatim |
| `verify_baseline` | the pre-edit outcome of the project's test command, recalled on a failing post-edit run |
| `bash_line_clip` | per-line cap so one minified line can't spend the whole output budget; the full text goes to a spill file |
| `edit_miss_diagnose` | a failed edit says whether the change looks *already applied*, or the first line/column where the sent text diverges |
| `trim_spill` | when compaction head/tail-trims an older tool result, the full original goes to a spill file and the trimmed message names the path |
| `result_spill` | when the per-result backstop cap clips a result, the full body goes to a spill file and the notice names the path |
| `rg_replace_flag_note` | one line naming what an `rg -r` result actually is: `-rn` is `--replace n`, not grep's recursive flag |

- `CHAD_DISABLE`: comma-separated lever names to switch off (`all` = every lever).

### Safety & A/B opt-outs

**One thing no permission mode waves through.** A `write`/`edit` whose real path is
outside the working directory (or under `.git/hooks`) always asks, even in auto-accept
and yolo; headless runs block it and tell the model why. The prompt names the resolved
path, so a symlink out of the workspace shows where the write actually lands.

These flip behavior off rather than tune it. The two safety opt-outs **weaken** chad's
defenses. Leave them unset in normal use; they exist for measurement and edge cases.

```bash
CHAD_NO_VALIDATE=1          uv run chad  # A/B knob: DISABLE arg coercion + schema validation
CHAD_NO_GOVERNOR=1          uv run chad  # A/B knob: DISABLE the runaway-turn governor
CHAD_NO_REPEAT_GUARD=1      uv run chad  # A/B knob: DISABLE the degenerate-repetition stop
CHAD_NO_SYNTAX_GATE=1       uv run chad  # A/B knob: DISABLE the post-edit syntax gate
CHAD_NO_PREFIX_CACHE=1      uv run chad  # measurement knob: drop the persistent prefix KV cache
CHAD_NO_SKILLS=1            uv run chad  # disable Agent Skill discovery (no /<skill>)
CHAD_NO_FASTPATH=1          uv run chad  # A/B knob: disable the fused-projection decode fast path
CHAD_NO_DESTRUCTIVE_GUARD=1 uv run chad  # DISABLE the catastrophic-bash screen (unsafe)
CHAD_NO_SEATBELT=1          uv run chad  # DISABLE the macOS Seatbelt sandbox for yolo bash (unsafe)
CHAD_NO_ENV_GUARD=1         uv run chad  # let bash children inherit credential-shaped env vars
CHAD_PROTECT_GIT=1          uv run chad  # also write-DENY .git inside the yolo sandbox
```

- `CHAD_NO_VALIDATE`: **disables** the typia-style lenient-parse → typed-validate →
  self-repair loop for tool-call arguments (`validate.py`), falling back to a strict
  `json.loads` plus a terse missing-required check. This *weakens* input handling (malformed
  or loosely-typed tool calls that chad would normally coerce/repair will instead error). An
  A/B knob to measure what validation buys per model. Leave unset in normal use.
- `CHAD_NO_GOVERNOR`: **disables** the runaway-turn governor (see [Turn budgets &
  think-cap](#turn-budgets--think-cap)), so a turn is never force-ended on its
  prefill/wall-clock budget. An A/B knob for measuring what the governor buys; the turn
  runs until the model stops on its own.
- `CHAD_NO_REPEAT_GUARD`: **disables** the degenerate-repetition stop (`guardrails.py`).
  Greedy decode on a small quantized model can lock into repeating one short string until
  the per-step token cap, minutes of dead generation per occurrence. By default chad
  watches the generation's tail, cuts the step off as soon as it turns fully periodic, and
  nudges the model out of the loop (aborting the turn after 3 cut-offs). Unlike the
  think-cap this never trades capability (it only fires on output that is already garbage),
  so it is on by default; this knob is the A/B arm.
- `CHAD_NO_SYNTAX_GATE`: **disables** the post-edit syntax gate (`syntaxgate.py`),
  which normally warns when an edit *introduces* a new syntax error (it never flags a
  pre-existing one). An A/B arm for evaluation runs; leave unset in normal use.
- `CHAD_NO_PREFIX_CACHE`: a fairness/measurement knob that **drops** the persistent
  prefix KV cache (`engine.py`), forcing a full re-prefill every step. It exists to measure
  what the cache is worth and makes chad much slower. Never set it in normal use.
- `CHAD_NO_DESTRUCTIVE_GUARD`: **disables** the catastrophic-bash screen
  (`guardrails.py`) even in `--yolo` mode. With it set, an injected `rm -rf ~`,
  `mkfs`, `dd of=/dev/…`, fork bomb, or `curl … | sh` is **not** screened before running.
  It is a screen, not a security boundary (the sandbox below is). Leave it unset.
- `CHAD_NO_SEATBELT`: **disables** the macOS Seatbelt sandbox (`seatbelt.py`) that
  yolo-mode bash commands run under by default: file writes confined to the workspace,
  temp dirs, and caches; reads and network open. Only the spawned shell child is ever
  sandboxed. Set this only when the sandbox itself breaks a legitimate workflow.
- `CHAD_NO_ENV_GUARD`: bash children normally get a **filtered** copy of the
  environment: variable names shaped like credentials are dropped — `…_TOKEN`,
  `…_SECRET`, `…_PASSWORD`, `…_API_KEY`, `…_KEY`, `…_PAT`, `…_AUTH`, `…_DSN`,
  `…_TOKEN_FILE`, plus `SSH_AUTH_SOCK`, `DATABASE_URL` and `AWS_PROFILE` by name. The
  one value the guard reads is a `…_URL` carrying userinfo (`scheme://user:pass@host`),
  dropped because the name gives no hint that it holds a password. Set this for a
  session whose commands legitimately need a credential (e.g. `gh`, deploy scripts) — a
  stripped variable is absent, never corrupted, so a command that needs one fails
  clearly.
- `CHAD_PROTECT_GIT`: an opt-in tier on top of the yolo sandbox: the workspace's
  `.git` (and a worktree's external gitdir) is write-DENIED, so an unreviewed command
  cannot destroy project history. The cost is real: every `.git`-writing git command
  (commit, add, checkout) EPERMs inside the sandbox, which is why it is opt-in.
- `CHAD_NO_SKILLS`: turns off [Agent Skill](#agent-skills-agentskillsio) discovery
  entirely, so no `/<skill>` command resolves. Skills no longer touch the system prompt,
  so this is no longer needed to keep your personal skills out of a benchmark; set it
  when you want them unreachable from chad at all. Like every other `CHAD_NO_*` flag it
  reads as set or unset: any non-empty value, `0` included, turns discovery off.
- `CHAD_NO_FASTPATH`: disables the fused-projection + compiled decode step installed
  at load for the dense `qwen3_5` hybrid (`mlx_fastpath.py`): the MLP `gate|up` concat, the
  GDN `in_proj` concat, and the compiled S=1 layer step; on the
  [shipped ternary weights](#what-the-engine-does-with-the-ternary-weights) also the
  `q|k|v` concat and the one-rotation-per-fused-matmul bodies. It is a silent no-op on any
  other checkpoint, so an arbitrary `--model` neither gains nor loses anything here. Pure speed,
  no behavior change, so this is an A/B and bisection knob rather than something to run
  with.
chad sets **no** `MLX_*` runtime variables, so there is nothing to opt out of:
`MLX_METAL_FAST_SYNCH`, `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` were each
measured end-to-end and every setting was *slower* than mlx's own defaults.
Export them yourself if you want to experiment; mlx reads them directly.

### Speculative decoding & kernel knobs

Everything in this block is speed only, bisection and A/B knobs, not something to
run with. Two different kinds of exactness live here, and the difference matters when you
are chasing a behaviour change:

- The speculation knobs are exact in the acceptance rule, not bit-identical in the
  forward. `CHAD_NO_DFLASH` and the width settings change *which* tokens get proposed;
  every emitted token is still the target's own choice for its position, and the sampled
  path keeps the model's true distribution. But a verified block is a batched S>1 forward
  and a serial step is an S=1 forward, and those run different matmul and attention
  kernels, so their logits agree to rounding, not to the bit: a greedy run therefore
  follows serial until the first near-tie and can take the other branch there. Measured
  on the shipped model (`benchmarks/spec_decode.py`, greedy, 10 repo-text seeds): 4/10
  160-token generations token-identical to serial, the rest diverging 10-95 tokens in at
  the same positions whether or not the cache is shared, and at *different* positions
  with `CHAD_NO_QMM_MMA=1`, the signature of rounding, not of a logic bug. A divergence
  that is not a near-tie (different output from the first token on an ordinary prompt,
  or a quality drop) is a bug.
- The kernel knobs are exact to rounding, not bit-identical. `CHAD_NO_QSDPA*` swap one
  attention kernel for another; each is within output-dtype rounding of an fp32 reference
  (the acceptance class MLX holds its own fused kernels to), but they are not bit-identical
  to *each other*. A greedy near-tie can therefore land on a different token and
  autoregression will amplify it into different prose. Measured: flipping
  `CHAD_NO_QSDPA_WIDE` on a 150-token greedy generation produced equally valid but
  differently worded output. That is expected, not a bug, so bisect a *behaviour* change
  with the speculation knobs, and read a kernel-knob output diff as noise unless the
  quality moves.

```bash
CHAD_NO_DFLASH=1          uv run chad  # disable block speculation (decode serially)
CHAD_DFLASH_DRAFT=7       uv run chad  # verified-width cap per round (1..7; default the full block)
CHAD_DFLASH_ADAPTIVE=0    uv run chad  # verify the full block every round instead of the per-round schedule
CHAD_DFLASH_PATH=/dir     uv run chad  # explicit drafter checkpoint or built sidecar dir
CHAD_USE_PLD=1            uv run chad  # OPT-IN: wide prompt-lookup decoding
CHAD_NO_QSDPA_WIDE=1      uv run chad  # disable the S>1 tier of the fused attention kernel
CHAD_NO_QSDPA_WIDE_SGM=1  uv run chad  # disable just its split-head variant
CHAD_QSDPA_WIDE_SGM_RT=1  uv run chad  # force the RT-split wide kernel instead of the one-read form
CHAD_NO_KERNEL_WARM=1     uv run chad  # skip warming verify-width attention kernels at load
CHAD_NO_QMM_MMA=1         uv run chad  # stock quantized_matmul at every verify width
CHAD_QMM_MMA_RECAL=1      uv run chad  # re-probe the small-M matmul kernel on this machine
```

- `CHAD_NO_DFLASH`: disables **DFlash2 block speculation** (`mlx_dflash.py`), the
  speculative path on the shipped model, and decodes one token per forward. A 1.9B drafter
  (an MLX port of [z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
  pre-quantized to 4-bit and shipped inside the model repo as `dflash/`) reads the main
  model's residual stream at five layers and proposes a whole block of tokens in **one**
  forward. The block is verified in one batched target forward and accepted by exact
  rejection sampling: every emitted token is the target's own choice and sampled output
  keeps the model's true distribution at any temperature (to kernel rounding; see the
  exactness note above: greedy runs follow serial until the first near-tie). Measured on an
  M4 Pro with the 3-bit quant, 10 prompts × 384-token decodes, medians: greedy
  serial 17.7 → **47.7 tok/s** (2.7×); at the thinking sampling preset (temp 1.0, top_p
  0.95, top_k 20) 22.4 → **41.4**; non-thinking 23.7 → 42-43; code 24 → 35-37. It engages
  when the drafter ships with the loaded weights: the shipped model and the 3-bit
  alternative both bundle it; a checkpoint of the same shape with no bundle (the upstream
  Prism pack) borrows the shipped model's (`mlx_dflash.DONORS`, keyed on hidden size, layer
  count and vocab); any other `--model` decodes serially unless you point
  `CHAD_DFLASH_PATH` at a drafter built for it.
- `CHAD_DFLASH_DRAFT` / `CHAD_DFLASH_ADAPTIVE`: the verified width. The drafter always
  proposes its full block of 7; by default a per-round schedule (a cost model over the
  measured round costs and recent acceptance) picks how many of those proposals to verify,
  narrowing or skipping a round when acceptance drops. That default was chosen on the
  **floor**, not the median: under the small-M matmul kernel below a full-block round
  costs ~2.2 serial steps whatever it commits, so on low-acceptance text a fixed block
  lands *below* serial decoding (worst measured prompts: 11.7 tok/s at the thinking
  preset against 15.2 serial; 17.5 greedy on a real session context where the schedule
  held 21.4), while the schedule never measured under serial. It pays for that with ~12%
  at the greedy median (31.7 vs 36.0 tok/s on real contexts) and ties at the thinking
  preset. `CHAD_DFLASH_ADAPTIVE=0` verifies the full block every round, the faster
  arm if you decode greedily on text the drafter knows. `CHAD_DFLASH_DRAFT=N` caps the
  width (and implies the fixed arm). Without the kernel (`CHAD_NO_QMM_MMA=1`) each extra
  verify row costs ~33 ms and width 4 is the optimum. `benchmarks/spec_decode.py` measures
  all three arms in one load, on this repo's text and on real mid-session contexts.
- `CHAD_DFLASH_PATH`: point the loader at a drafter outside the model repo: either an
  HF-layout checkpoint dir (quantized on first use into `~/.cache/chad/dflash/`) or an
  already-built sidecar dir. `python -m chad.mlx_dflash <dir> --out <model_dir>/dflash`
  builds a sidecar by hand; that is how the shipped repo's bundle was made.
- `CHAD_USE_PLD`: turns **wide prompt-lookup decoding** back on. It was the default
  before 2.0.0 and is now opt-in, because PLD drafts from *context recurrence* and can
  therefore only accelerate text that already appeared. On real agentic traces that is a
  minority of what this agent generates: ~62-66% of generated tokens are `<think>`, and
  reasoning prose replays at only ~2.3%. Whole-session contribution measured at **+2.2%**
  of generated tokens. It does not compose with block speculation (one generate loop each),
  so where the DFlash2 drafter is available it is strictly the better of the two, and
  this flag is how you measure the other arm.
- `CHAD_NO_QSDPA_WIDE` / `CHAD_NO_QSDPA_WIDE_SGM`: disable the multi-token (S>1) tier
  of the fused quantized-KV attention kernel (`mlx_qsdpa.py`), which serves speculative
  verification. Without it those steps fall back to dequantizing the whole cache once per
  attention layer, which the wide path beats by 8% at 8k context and 30% at 38k on the
  widths the adaptive schedule actually jumps to. Same numerics class as that
  fallback (both within output-dtype rounding of an fp32 reference), but see the
  note above: not bit-identical to it.
  (`CHAD_NO_QSDPA_WIDE_SGM` also lowers the width at which chad switches to that fallback,
  because the schedule left behind replays the S=1 kernel per row and so crosses over
  sooner.) Prefill chunks are far wider than any verify step and always take the
  fallback: flat in width, it is the faster of the two up there.
  (`CHAD_QSDPA_WIDE_KERNEL=1` forces the single-kernel variant instead of the split-head
  one, a kernel-selection knob for measurement.) The bigger hammers are still
  `CHAD_NO_QSDPA`, in [Safety & A/B opt-outs](#safety--ab-opt-outs), which disables the
  fused kernel entirely.
- `CHAD_QSDPA_WIDE_SGM_RT`: the wide tier has two forms. The default is *one-read*
  (K/V bytes touched once, all row tiles resident); this forces the *RT-split* form (one
  8-row tile per threadgroup, K/V read once per tile), which is what runs anyway at S>4.
  Both are exact; the knob exists to A/B them at a given width.
- `CHAD_NO_QMM_MMA` / `CHAD_QMM_MMA_RECAL`: the **small-M quantized matmul kernel**
  (`mlx_qmm_mma.py`) that serves speculative verification. Stock `quantized_matmul` is at
  roofline for one row and tiles well from ~13 rows, but in between (verify width = draft
  width + 1) its cost grows almost linearly in rows, because the GEMV path re-reads the
  weights per row (~33 ms per extra row on the shipped model). The MMA kernel (avlp12's
  `qmm_mma4` via mlx-dspark, MIT; with a width-generic unpack so the 3-bit body and 5-bit
  `lm_head` qualify) dequantizes each weight group once for all rows, so widths 6-8 cost
  about what width 5 does. At load chad probes every eligible weight shape on *this*
  chip × mlx version (numerics against the stock kernel, then a dependent-chain race at
  each width), and routes only the (shape, width) pairs that won; the verdict is cached
  under `~/.cache/chad/qmm_mma/`. Same exactness class as the attention-kernel knobs
  above: within rounding of the stock kernel, not bit-identical to it. Measured on the
  shipped model's shapes (dependent chains, M4 Pro): the 3-bit MLP matmuls 1.31× at six rows
  and 1.64× at eight, the 5-bit `lm_head` 1.55× / 1.87×, and below five rows the stock
  kernel wins, so verify widths 6-8 now cost about what width 5 does, which is what lets
  the block drafter verify its full block (see `CHAD_DFLASH_DRAFT`). Past one tile the
  kernel is **tiled**: a 9-24 row forward runs as 8-row kernel calls plus a remainder
  (through the kernel when it is wide enough to win, stock otherwise). The same probe
  races the tiled path against stock at 12, 16, 24, 32 and 48 rows per shape and keeps
  the widest that won (24 on every shipped shape, M4 Pro: 1.5-1.8× at 12-16 rows, 1.2× at
  24, a loss at 32, where mlx's own tiling takes over). `CHAD_NO_QMM_MMA=1` is the A/B
  arm; `CHAD_QMM_MMA_RECAL=1` re-probes (after an mlx upgrade, say).
  `benchmarks/verify_ladder.py` prints what a forward of each width costs on your machine.
- `CHAD_NO_KERNEL_WARM`: the attention kernel is templated on its verify width, so a
  width that has never run means a Metal compile lands on the critical path of a real
  step. Load warms exactly the widths *this* configuration can dispatch (the ones the
  draft schedule can pick, intersected with the ones the fused kernel serves) rather than
  the union of everything. Opting out moves those compiles into your first few steps.

### Dev & instrumentation

Not supported surface: they exist for debugging chad itself, and their formats can change
between releases.

```bash
CHAD_TRAJECTORY_JSON=/tmp/traj.json uv run chad  # record an ATIF trajectory (pure observer)
CHAD_SPILL_DIR=/tmp/spill           uv run chad  # where truncated tool output spills to disk
CHAD_DUMP_RENDER=/tmp/prompt.txt    uv run chad  # dump the fully-rendered prompt each step
CHAD_PREFILL_TRACE=/tmp/pf.jsonl    uv run chad  # per-step prefill/cache telemetry
CHAD_CHECKPOINT_DIR=/tmp/ckpt       uv run chad  # relocate the shadow-git edit checkpoints
CHAD_SESSION_DIR=/tmp/sessions      uv run chad  # relocate saved sessions (--continue/--resume)
```

- `CHAD_CHECKPOINT_DIR`: where the shadow-git repositories backing `/undo` and
  `/restore` live (default `~/.chad/checkpoints`, keyed per workspace). It exists so a test
  or eval suite never writes real home state. Note this is *not* `~/.chad/history`, which is
  the TUI's prompt-history file. The store is private (mode `0700`), never snapshots
  `.env*`, `*.pem`, `*.key` or SSH private keys, and a workspace's snapshots are swept
  after 30 days without an edit.
- `CHAD_SESSION_DIR`: where saved conversations live (default `~/.chad/sessions`, one
  directory per project) for `--continue`, `--resume` and the TUI `/resume` picker. Like
  `CHAD_CHECKPOINT_DIR`, it exists so a test or eval suite never touches real home state.

### Tree-sitter tags (ambient structure)

The `bash_read_skeleton` lever's one-line symbol maps and definition pointers come from
a tree-sitter tags index. Defaults are tuned for a big repo on a memory-tight machine;
you rarely need to touch this.

```bash
CHAD_REPOMAP_WORKERS=4   uv run chad  # subprocess workers for a cold repo scan (1 = serial)
```

- `CHAD_REPOMAP_WORKERS`: how many `python -c` subprocess workers a cold whole-repo
  tag scan shards across (`repomap.py`; default: cores−2, capped at 8). Workers import
  only `chad.repomap`, never the MLX engine. Tags persist per repo under
  `~/.chad/cache/repomap/` (mtime-validated per file), so warm sessions skip the scan.

### Voice mode (`/speech`)

Voice mode is all on-device: [Parakeet-on-MLX](https://github.com/ml-explore/mlx) (vendored)
transcribes your mic, macOS `say` speaks the replies. It needs the `speech` extra (see
[Installing](usage.md#installing--upgrading)), and nothing leaves the machine.

```bash
CHAD_VOICE="Daniel"        uv run chad  # macOS `say` voice (default: the system voice)
CHAD_SPEECH_RATE=200       uv run chad  # `say` rate in words/minute (default: the system rate)
CHAD_STT_QUANT=4           uv run chad  # ASR weight quantization: 8 (default), 4, or none
CHAD_STT_MODEL=<hf-repo>   uv run chad  # override the ASR checkpoint
CHAD_SPEECH_WORDS=/path.json uv run chad # personal word table (default ~/.chad/speech_words.json)
```

- `CHAD_VOICE`: an installed macOS voice name. An unknown name is refused at startup
  with a did-you-mean rather than silently falling back; `say -v '?'` lists what you have.
- `CHAD_STT_QUANT`: `8` by default. `4` halves the ASR weights again but is **opt-in**:
  clean-audio testing can't rule out degradation on a noisy mic. `none` keeps full precision.
- `CHAD_STT_MODEL`: defaults to `mlx-community/parakeet-tdt-0.6b-v3`. Anything else is
  unsupported; the vendored decoder is written to this model's output contract.
- `CHAD_SPEECH_WORDS`: the JSON word table that teaches the transcriber your jargon
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
files are created mode `0600`: they hold full tool arguments and results (see [Session
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
trace (the user query, tool-call args (including bash commands and write/edit content),
and result previews) to `~/.chad/session.log`. It's size-bounded (rotated, 5 MB × 3)
and passes previews through a best-effort secret redactor, but it still records
command/file previews in plaintext outside the repo, so treat it as sensitive.

**Privacy-first default: the trace is OFF.** chad is a local, single-user agent, so
nothing leaves your machine, but because the log lands plaintext previews under
`~/.chad`, it is opt-in. Set **`CHAD_SESSION_LOG=1`** (any truthy value) to turn it on;
the same flag also enables the persistent input history at `~/.chad/history` (mode
`0600`). When it's off, chad installs a null handler and won't create `~/.chad` for the
log's or history's sake.

**`CHAD_NO_SESSION_LOG=1`** remains a hard kill switch: if set it forces both the log
and the history off, and wins even when `CHAD_SESSION_LOG` is also set. (For the same
privacy reason, the resumable conversation store under `~/.chad/sessions/` (which holds
full tool args and results, and is written only when you use `-c`/`--resume`) is created
mode `0600`.)
