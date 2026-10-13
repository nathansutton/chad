# Configuration & reference

Steering chad (project instructions, Agent Skills, MCP servers, plan mode) and the full
flag and environment-variable reference. For the basics, see the [README](../README.md).

## Contents

- [Project instructions (CLAUDE.md / AGENTS.md)](#project-instructions-claudemd--agentsmd)
- [Agent Skills (agentskills.io)](#agent-skills-agentskillsio)
- [MCP servers (modelcontextprotocol.io)](#mcp-servers-modelcontextprotocolio)
- [Plan mode](#plan-mode)
- [Slash commands](#slash-commands)
- [Context window (agentic coding needs room)](#context-window-agentic-coding-needs-room)
- [Advanced (env vars)](#advanced-env-vars)
  - [The model](#the-model)
  - [Smoke test (`chad prove`)](#smoke-test-chad-prove)
  - [Alternate backend (remote)](#alternate-backend-remote)
  - [Sampling & reasoning effort](#sampling--reasoning-effort)
  - [Turn budgets & think-cap](#turn-budgets--think-cap)
  - [Harness levers](#harness-levers)
  - [Safety & A/B opt-outs](#safety--ab-opt-outs)
  - [Speculative decoding & kernel knobs](#speculative-decoding--kernel-knobs)
  - [Dev & instrumentation](#dev--instrumentation)
  - [Tree-sitter tags (ambient structure)](#tree-sitter-tags-ambient-structure)
  - [Voice mode (`/speech`)](#voice-mode-speech)
  - [Sessions](#sessions)
  - [Session log & privacy](#session-log--privacy)

## Project instructions (CLAUDE.md / AGENTS.md)

Standing instructions for a project (conventions, the commands you want used, things not
to touch) go in a markdown file at the root of that project. chad reads `CLAUDE.md`, or
`AGENTS.md` if there is no `CLAUDE.md` with content, from the **working directory only**:
not parent directories, not `~`. The two are never merged, and chad says at startup which
file it read and when a second one is ignored.

The first **4,000 characters** are used, so put what matters at the top; chad says when it
cut the file. It is read when a session starts, so an edit applies after `/reset` or in
the next session. The text is appended to the system prompt below the cache boundary with
the other per-project context, so it is prefilled once and restored from that project's
warm-start checkpoint, never re-sent per turn.

**`/init` writes one for you.** It reads whichever of `README`, `pyproject.toml`,
`package.json`, `go.mod`, `Cargo.toml` and `Makefile` exist and writes a concise file: an
overview, the main components, the real build/run/test commands, and any conventions
worth noting. It improves the file the project already has rather than shadowing it with
a new one, and the write goes through the usual confirmation in `normal` mode.

## Agent Skills (agentskills.io)

chad implements the open [Agent Skills](https://agentskills.io) format, so a skill
written for Claude Code works here unchanged: a folder with a `SKILL.md` (YAML frontmatter
with `name` and `description`, then markdown instructions), optionally bundling
`scripts/`, `references/` and `assets/`.

| Scope | Paths |
|---|---|
| Project | `./.agents/skills/`, `./.claude/skills/` |
| User | `~/.agents/skills/`, `~/.claude/skills/` |

Project skills override user skills on a name clash; a builtin always wins over both.

**You choose the skill, not the model.** Every installed skill is a slash command. Type `/`
and the completion menu lists them beside chad's builtins; `/skills` prints the same list
with discovery warnings. None of that reaches the model. `/ship` submits that one skill's
instructions as your turn, wrapped in `<skill>` with its directory and bundled files
listed, and anything after the command is the task: `/investigate the flaky test`. Bundled
files are read on demand from `bash`.

This diverges from the spec's tier-1 disclosure, where a catalog of every skill rides in
the system prompt for the model to pick from. That catalog measured 4,751 tokens for 62
installed skills, 60% of chad's system prompt, paid on every turn so a small model could
guess at a choice you can make from a menu. The cost you do pay is the skill body, which
can be most of the window: chad prints the token count when you load one
(`loaded skill ship (41,238 tokens)`).

Parsing is lenient: only a missing description or unparseable YAML skips a skill, and
anything else loads with a warning. A loaded skill's turn is exempt from compaction, and
re-running the command says it is already loaded instead of sending a second copy.

## MCP servers (modelcontextprotocol.io)

Agent Skills add instructions; **MCP** adds tools. chad connects to
[Model Context Protocol](https://modelcontextprotocol.io) servers over the official `mcp`
SDK, so it speaks both **stdio** (local subprocesses) and **Streamable HTTP** (hosted).

| Scope | Path |
|---|---|
| Project | `./.mcp.json` (the Claude Code convention) |
| User | `~/.chad/mcp.json` |

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

A `url` means Streamable HTTP, otherwise `command` means stdio; `type` is accepted but
the key present is authoritative. `headers` are sent on every request, `"disabled": true`
skips a server, `"timeout"` (seconds) overrides the per-call limit and `"connect_timeout"`
bounds the connect. On the first turn chad connects the eligible servers in parallel and
time-bounded, so one dead endpoint cannot stall the others, and prints one line saying
how many connected, failed, or are waiting for `/mcp trust` or `/mcp login`.

**Project trust.** A `./.mcp.json` does nothing until you run `/mcp trust` in that
directory. A project file may have just been cloned, and a stdio server is an arbitrary
local command, so an untrusted repo must not launch a subprocess the moment you open chad
in it. Until then `/mcp` shows each gated server with what it would do if trusted: the
command and arguments, or the URL, with header and `env` names but never values.
`/mcp trust` records the project's absolute path in `~/.chad/trusted_mcp.json` (mode
`0600`) and lists what it trusted; moving the repo re-prompts, and a directory with no
`.mcp.json` trusts nothing. User-scope servers are yours and auto-connect. A project
server with the same name as one of yours replaces it in that project, with a warning.

**Environment.** A stdio server inherits a minimal env allowlist (`PATH`, `HOME`, locale,
`TMPDIR`, `SHELL`, `USER`), not chad's environment: a local command has no business
inheriting your API keys. A server that needs a variable declares it in its `env:` block.
`CHAD_MCP_FULL_ENV=1` restores the full inherit, for every secret to every server, so
leave it unset unless you know which server needs what.

**OAuth (hosted connectors).** Static bearer tokens via `headers` work out of the box.
Connectors that require OAuth (Linear, Slack, Atlassian) are behind `CHAD_MCP_OAUTH=1`
and `"auth": "oauth"` on the server:

```json
{ "mcpServers": { "linear": { "type": "http", "url": "https://mcp.linear.app/mcp", "auth": "oauth" } } }
```

The first login is interactive, so OAuth servers never auto-connect; they show as
`needs login` in `/mcp` until you run `/mcp login linear`. That opens the browser (or
prints the URL), catches the redirect on a loopback server, and stores the tokens in
`~/.chad/mcp_tokens.json` (mode `0600`); later runs reconnect without you. A login is
tied to the URL it was made for, so a project server reusing one of your names but
pointing elsewhere needs its own. With the flag unset, an `auth: oauth` server is skipped
with a warning. A headless session is never blocked: a server that cannot log in
contributes no tools.

**In the harness.** Server tools are namespaced `mcp__<server>__<tool>` and go through the
same validate-and-repair path as the builtins, driven by each tool's `inputSchema`. A
tool the server marks `readOnlyHint` runs without a prompt; every other MCP tool asks
first and shows its arguments, and in plan mode only read-only tools run. A server that
is missing, slow, or crashes mid-session contributes no tools and never takes the agent
down. `/mcp` shows each server's transport, status, tools and warnings.

## Plan mode

`--plan`, or shift-tab round to `plan mode`, makes the session read-only with one
exception: `write` and `edit` are allowed when the path resolves inside **`./plans/`**.
Every other mutating tool, `bash` included, is refused with a note telling the model to
investigate read-only and write its plan instead. A plan write does not ask for
confirmation, so plan mode refuses to replace a plan file this session did not write.

A change request is answered with one self-contained `plans/NNN-kebab-title.md`
(continuing the number sequence already there): context, file paths with excerpts,
numbered steps, verify commands, and what is out of scope. `./plans/` is an ordinary
directory in your repo; commit it or ignore it. A plain question is answered in prose.

When a plan-mode turn finishes having written or edited a plan, chad prints
`plan ready → <path>` and waits. Type to steer, or press **ctrl-g** or run **`/accept`** to
accept it: the context is cleared, the permission mode goes back to the one chad started
in, and a fresh turn reads the file and executes each step, running the verification
commands at the end. `/accept <path>` accepts any plan file, including one from an earlier
session, and `/accept` with nothing pending lists the newest plans. A plan left
un-accepted is just a file on disk.

## Slash commands

Typed in the TUI (`/` opens a completion menu with these and every installed skill). Most
cost nothing in context; `/init` and `/accept` start a real turn. The plain `--repl` front
end runs the ones marked in the REPL column. A command that does not exist is reported
with a suggestion, not sent to the model.

| Command | What it does | REPL |
|---|---|---|
| `/help` | commands & keybindings | ✓ |
| `/init` | analyze the project and write its instructions file (a real turn) | ✓ |
| `/skills` | list installed Agent Skills with discovery warnings | ✓ |
| `/mcp` | MCP server status; `/mcp trust` trusts this project's servers; `/mcp login <server>` authenticates one | ✓ |
| `/compact` | reclaim context now | ✓ |
| `/ctx` | where the context window is going, in tokens | – |
| `/undo` | revert files to the last edit checkpoint, snapshotting the current state first so `/undo` again brings it back | – |
| `/restore` | list edit checkpoints; `/restore <hash>` reverts to one | – |
| `/resume` | list recent sessions; `/resume <n>` forks one | – |
| `/reset`, `/clear` | clear the conversation and the KV cache | ✓ |
| `/model` | show model and context window | ✓ |
| `/mode` | cycle the permission mode | ✓ |
| `/speech` | toggle voice mode | – |
| `/accept [path]` | accept the pending plan, or the named plan file, and implement it | – |
| `/exit`, `/quit` | quit | ✓ |

## Context window (agentic coding needs room)

chad uses the model's full native window, 262k on the shipped model, and lets the
RAM-aware governor decide how much of it this machine can spend. The KV cache grows
lazily, so a large window costs nothing until tokens fill it, and it is quantized to
8-bit by default, which halves the footprint and is faster than fp16 at long context
because the fused attention kernel is written for it.

Qwen3.8-27B is a hybrid: 48 of its 64 layers carry a fixed-size recurrent state, so only
the 16 attention layers grow with context, at **34,816 bytes per token** (a pure-attention
transformer of the same shape would spend ~4× that):

| Context | KV cache (8-bit) | |
|---|---|---|
| 32k | 1.1 GB | |
| ~74k | 2.6 GB | where a 24 GB Mac lands on the shipped `UD-Q3_K_XL` |
| 128k | 4.6 GB | |
| ~138k | 4.8 GB | where it lands on `UD-IQ3_XXS` (`--model`) |
| 262k | 9.1 GB | native; unreachable on 24 GB |

The compaction threshold is sized from the live Metal budget minus the resident weights,
a headroom band and the prefill transient (flat at ~2 GB on the default cache, ~4.3 GB on
fp16), then capped at the window. On 24 GB the weights (13.2 GB), drafter (1.2 GB) and
transient spend ~86% of the budget before the first cached token, which is why it lands
near ~74k. The banner states the window you actually got, and says when the governor
bound it: `74k of 262k context`. The threshold also respects host free memory (pressure
from Docker or other apps the Metal budget cannot see) and is re-checked between turns.

At load the engine wires the Metal working set and caps the allocator slightly below it,
so a transient spike back-pressures instead of becoming a jetsam kill; a Metal OOM inside
a prefill chunk halves the chunk and retries.

```bash
CHAD_MAX_CONTEXT=131072   chad  # cap the window; for an arbitrary --model, YaRN-extends past native
CHAD_KV_BITS=0            chad  # fp16 KV cache (roughly doubles the table above)
CHAD_CTX_LIMIT=28000      chad  # force the compaction threshold (overrides the governor)
CHAD_CTX_SAFETY=0.95      chad  # fraction of the Metal budget the sizing may spend (default 0.975)
CHAD_CTX_SLOPE_FACTOR=1.5 chad  # A/B: per-token cost multiplier for the sizing (default 1.0)
CHAD_KV_CACHE_MAX_GB=4    chad  # disk budget for KV checkpoints (default 8; 0 = unlimited)
CHAD_PREFILL_CHUNK=1024   chad  # fixed prefill chunk (default adaptive: 512, decaying to 256 under pressure)
CHAD_NO_MEMORY_CLAMP=1    chad  # A/B: skip the Metal allocator clamps
CHAD_NO_QSDPA=1           chad  # debug: quantized cache without the fused kernel (the old slow path)
```

`CHAD_CTX_SAFETY` is the one headroom lever: lower it if you run memory-hungry apps
beside chad. `CHAD_KV_CACHE_MAX_GB` is the only one that spends disk: `~/.cache/chad/kv`
holds one warm-prefix checkpoint per distinct system prompt (a ~51 MB floor each, from
the recurrent state) plus the whole-session checkpoints a quit writes (~35 KB per token
of conversation plus 150 MB), LRU-evicted down to the budget after each write.

## Advanced (env vars)

A `CHAD_*` switch is on when the variable is set to anything at all, `0` included; to turn
one off, unset it. Exceptions: `CHAD_MCP_FULL_ENV` must be exactly `1`, and
`CHAD_DFLASH_ADAPTIVE` and `CHAD_MCP_OAUTH` read `0` or `false` as off. A memory-sizing
variable that does not parse stops the run with the variable, the value and the fix; the
budget knobs say once on stderr when a value is ignored.

### The model

chad ships one: Qwen3.8-27B (a dense `qwen3_5` hybrid, 64 layers, 262k native context) as
Unsloth's [`UD-Q3_K_XL`][model] GGUF, ~13.2 GB resident. chad reads the GGUF natively: the
llama.cpp blocks load unchanged and chad's Metal kernels decode them, bit-exact to
llama.cpp's dequantizers. [Design](design.md#the-weights) explains why.

The tokenizer, chat template and 4-bit DFlash2 drafter come from a 1.2 GB sidecar,
[`nathansutton/Qwen3.8-27B-DFlash2-MLX`][sidecar], which holds no target weights. Its
template renders `reasoning_effort=medium`. The first start converts the GGUF into MLX
arrays (same blocks, same bytes) under `~/.cache/chad/gguf/`: ~13 GB more disk, ~75 s
once, ~9 s every start after. A new upstream revision gets a fresh directory and the
stale one is deleted.

`--model` (or `CHAD_MODEL`; the flag wins) takes `auto`, a Hugging Face repo id, a local
model dir, or a GGUF as a `.gguf` path or `owner/repo/file.gguf`. There are no size
shorthands. Other weights run through the same engine and lose speed, not correctness:
the drafter, the fused attention, the fast path and the governor's per-token cost each
decline to install or fall back to a stock path. The other Unsloth files of this model
have been measured, as has an MLX repack of the default's bit map:

| weights | resident | window, 24 GB | [polyglot][runs], 3 tasks × 3 |
|---|---|---|---|
| `UD-Q3_K_XL` GGUF (default) | 13.2 GB | ~74k | 9/9 |
| `UD-IQ3_XXS` GGUF | 10.9 GB | ~138k | 8/9 |
| [affine 3-bit repack][q3] | 12.3 GB | — | 6/9 |

The affine repack is chad's MLX quantization using the bit map of Unsloth's file, not its
recipe; the table is why it lost. Every run is in the public
[dataset of eval runs](https://huggingface.co/datasets/nathansutton/chad-polyglot-runs).

[model]: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
[sidecar]: https://huggingface.co/nathansutton/Qwen3.8-27B-DFlash2-MLX
[q3]: https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX
[runs]: ../benchmarks/polyglot/RUNS.md

**chad targets 24 GB Apple Silicon and nothing smaller.** On a smaller box it warns once
and runs anyway, with the window shrunk toward its floor.

### Smoke test (`chad prove`)

Downloads this machine's default model if it is not cached, then drives four tiny fix-it tasks
end-to-end: a real agent loop, real edits in a scratch directory, each verified by a check
script rebuilt from read-only sources so an edit cannot spoof a pass. It prints what
worked with time-to-first-token, decode speed and wall clock per task, and writes
`results.json` to the invoking directory.

It pins that default model, ignoring `--model` and `CHAD_MODEL`, because the question is
whether the thing you are about to run works on this machine. Once the weights are present
it goes offline (`HF_HUB_OFFLINE` plus a socket guard). Exit codes: `0` all passed, `1` a
task failed, `2` the preflight stopped. On working hardware it should never fail, so a
failure points at the install or the machine, not the model.

### Alternate backend (remote)

`--backend llama` drives the same harness against a remote llama.cpp server's raw
`/completion` endpoint with token-id prompts. It is the arm used when chad runs in a Linux
benchmark container against a GGUF served elsewhere, where MLX cannot run. The KV cache
lives in the server, so warm-prefix checkpoints are no-ops, but cache telemetry is real
and `<think>` comes back verbatim. It is not a general "use a cloud model" path.

```bash
chad --backend llama --base-url http://<host>:8081   # or CHAD_LLAMA_BASE_URL
```

- `--base-url` / `CHAD_LLAMA_BASE_URL`: the llama-server origin, no `/v1`. Required.
- `--tokenizer` / `CHAD_TOKENIZER`: an HF repo whose tokenizer matches the served
  model's vocab. Required for a GGUF server, since GGUF repos ship no tokenizer.
- `--api-key-env NAME`: the name of the env var holding the API key; the key is never
  passed on the command line.

### Sampling & reasoning effort

chad samples with the model card's recipe for the mode it is in. With reasoning on (the
default): temperature 1.0, top-p 0.95, top-k 20. With `--no-think`: temperature 0.7,
top-p 0.80, top-k 20. A variable below overrides that one setting and leaves the rest in
place. `CHAD_TEMP=0` is greedy and reproducible; its known failure is that a stalled or
garbled step replays itself byte for byte on every retry.

```bash
CHAD_TEMP=0.7             chad  # temperature (default 1.0; 0.7 with --no-think)
CHAD_TOP_P=0.95           chad  # nucleus sampling (default 0.95; 0.80 with --no-think)
CHAD_TOP_K=20             chad  # top-k (default 20)
CHAD_MIN_P=0.05           chad  # min-p tail trim (default 0 = off)
CHAD_PRESENCE_PENALTY=0.5 chad  # flat penalty on already-generated tokens (default 0)
CHAD_REASONING_EFFORT=low chad  # xhigh | medium | low, passed to the chat template
```

- `CHAD_MIN_P`, `CHAD_TOP_P`, `CHAD_TOP_K` trim the sub-noise-floor logit tail without
  touching temperature, which is usually what you want when a small quant invents an API.
- `CHAD_PRESENCE_PENALTY` ships at 0 and is worth leaving there. Model cards suggest up to
  1.5 for chat, but code must reuse identifiers, keywords and punctuation: measured at
  1.5, the model spent 45 steps exploring, landed zero edits, and emitted corrupted tool
  arguments.
- `CHAD_REASONING_EFFORT` is a template-level request on checkpoints whose template
  accepts one (Qwen3.8); unset, the argument is not passed at all. It is distinct from the
  harness-level [think-cap](#turn-budgets--think-cap), which force-closes a `<think>` the
  model has already started.

### Turn budgets & think-cap

A runaway-turn **governor** ends a turn that burns a lot of prefill without landing and
verifying a change: it nudges at ~50% of budget and at ~80% banks a one-line progress
note and stops. The token budget applies to every session and defaults to three times the
context limit; the wall-clock budget is off unless set. In the TUI the next thing you type
after a stop starts a fresh context seeded with the note; `/reset` starts clean.

These are env-only because the thing that sets them is an unattended harness, which
already builds a `CHAD_*` environment. `--think-budget` is the one kept on the CLI.

```bash
CHAD_THINK_BUDGET=1500        chad  # soft-cap each step's <think> at N tokens, force-close it and END the step
CHAD_THINK_CEILING=384        chad  # force-close a runaway <think> but keep decoding the action in the SAME step
CHAD_TURN_BUDGET_TOKENS=90000 chad  # governor token budget (default 3× the context limit)
CHAD_TURN_BUDGET_S=600        chad  # wall-clock variant, seconds (off by default)
CHAD_AUTO_CONTINUE=2          chad  # on a hard stop, relaunch a fresh turn seeded with the note, N times (default 2 for a yolo one-shot, else 0)
CHAD_REVIEW_PASS=1            chad  # a one-shot that finishes early spends the slack verifying (needs CHAD_TURN_BUDGET_S)
CHAD_MAX_GEN_TOKENS=32768     chad  # hard per-step generation cap (default 32768)
```

- `CHAD_THINK_CEILING` is the one to reach for first. It force-closes the runaway block and
  keeps decoding the action in the same step, so the reasoning so far stays in context and
  nothing is re-derived. Off by default: force-closing `</think>` is the most invasive
  thing the harness can do to the token stream, and the measured record says the bare
  loop does not need it.
- `CHAD_THINK_BUDGET` (`--think-budget`) ends the step instead, so the next step re-derives
  its reasoning, and escalates the cap when the turn keeps getting capped. When it fires,
  the status line shows a `✂N` counter.
- `CHAD_MAX_GEN_TOKENS` is a backstop against non-repetitive runaway garble, not a
  reasoning lever (a literal decode loop is the repeat guard's job). It is high on purpose:
  a low cap ends a long chain of thought inside `<think>` as discarded reasoning with no
  action.
- `CHAD_NO_GOVERNOR=1` disables the governor entirely (below).

### Harness levers

chad ships **ten** result-channel levers, all on by default, the survivors of a
measurement campaign in which nothing else beat the bare model and tool loop. Each makes
the `bash` route more honest or more informative, and each keeps a name and a switch for
one reason: leave-one-out ablation. A name that is not registered is a startup error,
since a typo would otherwise run the unmodified harness and report no effect.

```bash
chad levers                          # the registry, and what is active (loads no model)
CHAD_DISABLE=bash_line_clip chad     # leave-one-out arm
CHAD_DISABLE=all chad                # the bare model + tool loop
```

| Lever | What it adds to the result channel |
|---|---|
| `env_manifest` | the toolchain on this machine, with versions, and the common ones missing, in the system prompt tail |
| `bash_read_skeleton` | a one-line symbol map the first time a source file comes back through `cat`/`sed`/`head`, and a definition pointer when a grep for a known symbol is empty |
| `bash_empty_diagnose` | why a command printed nothing: a `sed` range past EOF, or the pipeline stage that matched nothing |
| `bash_trim_keep_failures` | when long output is head/tail-trimmed, the failure rows from the middle are kept verbatim |
| `verify_baseline` | the pre-edit outcome of the project's test command, recalled on a failing post-edit run |
| `bash_line_clip` | per-line cap so one minified line cannot spend the output budget; the full text goes to a spill file |
| `edit_miss_diagnose` | a failed edit says whether the change is already applied, or the first line and column where the sent text diverges |
| `trim_spill` | when compaction trims an older result, the original goes to a spill file and the notice names the path |
| `result_spill` | when the per-result cap clips a result, the full body goes to a spill file and the notice names the path |
| `rg_replace_flag_note` | one line saying what an `rg -r` result is: `-rn` is `--replace n`, not grep's recursive flag |

### Safety & A/B opt-outs

**One thing no permission mode waves through.** A `write` or `edit` whose real path is
outside the working directory, or under `.git/hooks`, always asks, even in yolo; headless
runs block it and tell the model why. The prompt names the resolved path, so a symlink
out of the workspace shows where the write lands.

These flip behaviour off rather than tune it. The ones marked unsafe weaken chad's
defences; leave them unset outside measurement.

```bash
CHAD_NO_VALIDATE=1          chad  # A/B: no arg coercion or schema validation; strict json.loads only
CHAD_NO_GOVERNOR=1          chad  # A/B: no runaway-turn governor
CHAD_NO_REPEAT_GUARD=1      chad  # A/B: no degenerate-repetition stop
CHAD_NO_SYNTAX_GATE=1       chad  # A/B: no post-edit syntax warning
CHAD_NO_PREFIX_CACHE=1      chad  # measurement: full re-prefill every step (much slower)
CHAD_NO_KV_RESUME=1         chad  # a quit writes no KV checkpoint; a resume re-reads the transcript
CHAD_NO_SKILLS=1            chad  # no Agent Skill discovery, so no /<skill> resolves
CHAD_NO_FASTPATH=1          chad  # A/B: no fused-projection decode fast path (speed only)
CHAD_NO_DESTRUCTIVE_GUARD=1 chad  # unsafe: no screen for rm -rf ~, mkfs, dd of=/dev/…, curl | sh
CHAD_NO_SEATBELT=1          chad  # unsafe: no macOS Seatbelt sandbox for yolo bash
CHAD_NO_ENV_GUARD=1         chad  # bash children inherit credential-shaped env vars
CHAD_PROTECT_GIT=1          chad  # also write-deny .git inside the yolo sandbox
```

- **The repeat guard** watches the generation's tail, cuts the step off as soon as it turns
  fully periodic, and nudges the model out of the loop, aborting the turn after three
  cut-offs. It only fires on output that is already garbage, so it never trades capability.
- **The destructive-bash screen** is a screen, not a security boundary; the sandbox is.
- **The yolo sandbox** confines file writes to the workspace, temp dirs and caches; reads
  and network are open. Only the spawned shell is sandboxed, never chad itself, which needs
  Metal. When yolo cannot be sandboxed, because this is set or chad is itself inside a
  sandbox, chad says so on entering yolo. `CHAD_PROTECT_GIT` adds a write-deny on `.git`,
  at the cost of every `.git`-writing git command failing inside the sandbox.
- **The env guard** drops variables named like credentials (`…_TOKEN`, `…_SECRET`,
  `…_PASSWORD`, `…_API_KEY`, `…_KEY`, `…_PAT`, `…_AUTH`, `…_DSN`, `…_TOKEN_FILE`, plus
  `SSH_AUTH_SOCK`, `DATABASE_URL` and `AWS_PROFILE`) and any `…_URL` carrying
  `user:pass@`. A stripped variable is absent, never corrupted: most tools fail with their
  own auth error, and the AWS CLI falls back to the default profile. chad lists the
  withheld names when an interactive session starts. A command you type with `!` is not
  filtered.
- **`CHAD_NO_FASTPATH`** is a no-op on any checkpoint the fast path does not cover; it is
  a bisection knob, not something to run with.

chad sets no `MLX_*` runtime variables: `MLX_METAL_FAST_SYNCH`, `MLX_MAX_OPS_PER_BUFFER`
and `MLX_MAX_MB_PER_BUFFER` were each measured end-to-end and every setting was slower
than mlx's defaults. Export them yourself to experiment.

### Speculative decoding & kernel knobs

Everything here is speed only: bisection and A/B knobs. Two kinds of exactness live here,
and the difference matters when you are chasing a behaviour change:

- The **speculation** knobs are exact in the acceptance rule, not bit-identical in the
  forward. Every emitted token is the target's own choice for its position, but a verified
  block is a batched S>1 forward and a serial step is S=1, and those run different kernels
  whose logits agree to rounding. A greedy run follows serial until the first near-tie and
  can take the other branch there (measured: 4 of 10 160-token greedy generations
  token-identical to serial, the rest diverging 10–95 tokens in). A divergence that is not
  a near-tie, or a quality drop, is a bug.
- The **kernel** knobs swap one attention or matmul kernel for another. Each is within
  output-dtype rounding of an fp32 reference, but not bit-identical to each other, so a
  greedy near-tie can land on a different token and autoregression amplifies it into
  different prose. Read a kernel-knob output diff as noise unless the quality moves.

```bash
CHAD_NO_DFLASH=1          chad  # decode serially
CHAD_DFLASH_DRAFT=7       chad  # verified-width cap per round (1..7; default the full block)
CHAD_DFLASH_ADAPTIVE=0    chad  # verify the full block every round instead of the per-round schedule
CHAD_DFLASH_PATH=/dir     chad  # a drafter checkpoint or built sidecar outside the model repo
CHAD_NO_SCR=1             chad  # compaction re-reads everything after the first change
CHAD_SCR_MAX_SPANS=6      chad  # most surviving spans a compaction moves instead of reads
CHAD_SCR_MIN_SPAN=64      chad  # shortest survivor worth moving
CHAD_USE_PLD=1            chad  # opt-in: prompt-lookup decoding (does not compose with the drafter)
CHAD_NO_QSDPA_WIDE=1      chad  # no S>1 tier of the fused attention kernel
CHAD_NO_QSDPA_WIDE_SGM=1  chad  # no split-head variant of that tier
CHAD_QSDPA_WIDE_SGM_RT=1  chad  # force the RT-split wide kernel instead of the one-read form
CHAD_QSDPA_WIDE_KERNEL=1  chad  # force the single-kernel wide variant
CHAD_NO_PREFILL_SLICE=1   chad  # prefill attention as one slab (the transient grows; smaller window)
CHAD_NO_KERNEL_WARM=1     chad  # skip warming the verify-width kernels at load (compiles land in your first steps)
CHAD_NO_QMM_MMA=1         chad  # stock quantized_matmul at every verify width
CHAD_QMM_MMA_RECAL=1      chad  # re-probe the small-M matmul kernel on this machine (after an mlx upgrade)
```

- **DFlash2** (`mlx_dflash.py`): a 1.9B drafter, an MLX port of
  [z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) pre-quantized to
  4-bit in the sidecar, reads the main model's residual stream at five layers and
  proposes a block of 7 tokens in one forward, verified in one batched target forward. By
  default a per-round schedule over the measured round costs and recent acceptance picks
  how many to verify, so low-acceptance text never pays for a block it mostly rejects;
  `CHAD_DFLASH_ADAPTIVE=0` verifies the full block every round, the faster arm on text the
  drafter knows well. Any checkpoint of this model's shape borrows the sidecar's drafter;
  any other `--model` decodes serially unless `CHAD_DFLASH_PATH` points at one built for it
  (`python -m chad.mlx_dflash <dir> --out <model_dir>/dflash` builds a sidecar).
  `benchmarks/spec_decode.py` measures the arms in one load.
- **Suffix reuse** (`suffix_reuse.py`): after compaction, the attention rows of text that
  survived are moved to their new positions and only the inserted text is read; the
  recurrent layers continue from whichever held state has just read the text at that
  point. At most `CHAD_SCR_MAX_SPANS` survivors are moved, each at least
  `CHAD_SCR_MIN_SPAN` tokens; survivors are found as runs of eight tokens, so a transcript
  of one short line repeated thousands of times is re-read rather than matched slowly. The
  prefill trace records the moved count per step as `relocated_tokens`.
- **Prompt-lookup decoding** was the default before 2.0.0. It drafts only from text that
  already appeared in context, which on real agent traces is ~2.2% of generated tokens,
  and it does not compose with the block drafter.
- **The wide attention tier** (`mlx_qsdpa.py`) serves speculative verification; without
  it those steps dequantize the whole cache per layer, which the tier beats by 8% at 8k
  context and 30% at 38k. Prefill chunks are far wider than any verify step and always
  take the fallback. `CHAD_NO_QSDPA` (above) disables the fused kernel entirely.
- **The small-M matmul kernel** (`mlx_qmm_mma.py`, avlp12's `qmm_mma4` via mlx-dspark,
  MIT) dequantizes each weight group once for all rows where the stock GEMV re-reads it
  per row, so verify widths 6–8 cost about what width 5 does, and a 9–24 row forward
  runs as tiled 8-row calls. At load chad probes every eligible shape on this chip and
  mlx version, routes only the (shape, width) pairs that won, and caches the verdict under
  `~/.cache/chad/qmm_mma/`. `benchmarks/verify_ladder.py` prints what each width costs on
  your machine.

### Dev & instrumentation

Not supported surface: formats can change between releases.

```bash
CHAD_TRAJECTORY_JSON=/tmp/traj.json chad  # record an ATIF trajectory (pure observer)
CHAD_PREFILL_TRACE=/tmp/pf.jsonl    chad  # one JSON row per prefill: tokens, seconds, render/compact/tool overhead
CHAD_DUMP_RENDER=/tmp/prompt.txt    chad  # dump the fully-rendered prompt each step
CHAD_SPILL_DIR=/tmp/spill           chad  # where truncated tool output spills to disk
CHAD_CHECKPOINT_DIR=/tmp/ckpt       chad  # relocate the shadow-git edit checkpoints
CHAD_SESSION_DIR=/tmp/sessions      chad  # relocate saved sessions
```

The last two exist so a test or eval suite never writes real home state. The checkpoint
store (default `~/.chad/checkpoints`, mode `0700`) never snapshots `.env*`, `*.pem`,
`*.key` or SSH private keys, and a workspace's snapshots are swept 30 days after its last
edit.

### Tree-sitter tags (ambient structure)

The `bash_read_skeleton` lever's symbol maps come from a tree-sitter tags index, persisted
per repo under `~/.chad/cache/repomap/` and validated per file by mtime, so warm sessions
skip the scan.

```bash
CHAD_REPOMAP_WORKERS=4   chad  # subprocess workers for a cold repo scan (default cores−2, capped at 8)
```

### Voice mode (`/speech`)

All on-device: Parakeet-on-MLX transcribes your mic, macOS `say` speaks the replies. Needs
the `speech` extra ([installing](usage.md#installing--upgrading)).

```bash
CHAD_VOICE="Daniel"          chad  # macOS `say` voice (default: the system voice; `say -v '?'` lists them)
CHAD_SPEECH_RATE=200         chad  # `say` rate in words/minute
CHAD_STT_QUANT=4             chad  # ASR weight quantization: 8 (default), 4, or none
CHAD_STT_MODEL=<hf-repo>     chad  # the ASR checkpoint (default mlx-community/parakeet-tdt-0.6b-v3; others unsupported)
CHAD_SPEECH_WORDS=/path.json chad  # personal word table that teaches it your identifiers (default ~/.chad/speech_words.json)
```

An unknown voice name is refused at startup with a did-you-mean. `CHAD_STT_QUANT=4` halves
the ASR weights again but is opt-in: clean-audio testing cannot rule out degradation on a
noisy mic.

### Sessions

Every conversation is saved per working directory, so `-c` in a project resumes that
project's thread and nothing else:

```
~/.chad/sessions/<cwdhash>/<session_id>.json   one file per session, mode 0600
~/.chad/sessions/<cwdhash>/index.json          title / last-updated / turn count
```

The newest **20** sessions per directory are kept, and chad says when a save removes one.
Resuming prints the session's title and the last things you asked.

```bash
chad -c            # resume this directory's most recent session
chad --resume      # list recent sessions, pick one by number (needs a TTY)
```

**Resuming forks; it never overwrites.** Both flags seed a fresh conversation with the old
messages under a new `session_id`, so the session you resumed from is left as it was and
you can branch from the same point twice.

**A resumed session does not re-read its conversation.** Quitting the TUI (`/exit`,
ctrl-d, a second ctrl-c) or the REPL also writes the engine's cache to `~/.cache/chad/kv`.
`chad -c`, `--resume` and `/resume` restore that file instead of prefilling the transcript:
well under a second, against about a second per hundred tokens on the shipped model, and
bit-identical to the cache the session ended with. The resumed session keeps the system
prompt it ran under; `/reset` starts a fresh one. A session that was not quit cleanly,
whose checkpoint was evicted from the disk budget, or that ran on another model, cache
mode or window resumes cold. The banner says which happened: `[resumed warm: N tokens of
the conversation restored from disk]`. `CHAD_NO_KV_RESUME=1` turns it off.

The facts the [result channel](#harness-levers) gathers (files edited, the test command's
outcome before the first edit, files already shown a symbol map) are saved with the
conversation and come back with it.

### Session log & privacy

The diagnostics log is **off by default**. When enabled, each session appends throughput
numbers and a readable trace (your query, tool-call arguments including bash commands and
edit content, result previews) to `~/.chad/session.log`, size-rotated (5 MB × 3) and
passed through a best-effort secret redactor. It still records previews in plaintext
outside the repo, so treat it as sensitive.

`CHAD_SESSION_LOG=1` turns it on, along with the persistent input history at
`~/.chad/history` (mode `0600`). `CHAD_NO_SESSION_LOG=1` is a hard kill switch that wins
even when both are set. The session store under `~/.chad/sessions/` holds full tool
arguments and results for every session and is created mode `0600` for the same reason.
