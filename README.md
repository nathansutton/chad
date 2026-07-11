# chad — a local, MLX-backed coding agent

[![tests](https://github.com/nathansutton/chad/actions/workflows/tests.yml/badge.svg)](https://github.com/nathansutton/chad/actions/workflows/tests.yml)

<img src="docs/tbench-size-vs-score.png" width="840" alt="Terminal-Bench 2.0: accuracy vs. model size. chad + Ornith (a 35B MoE) lands on the Claude Sonnet 4.5 reference line, matching open models many times its size and standing alone in the laptop-class ≤40B band.">


> Claude can do anything, for anyone, anywhere. chad does one thing. 🗿    
> *Coding under supervision.*

> The chart is the leaderboard's story. For your own machine's story:
> **`uvx chad-code prove`** — a two-minute, offline smoke test where chad fixes
> four small tasks in front of you, mechanically verified and timed. (It's a
> proof of life, not a benchmark — the tasks come from chad's own dev suite.)

## Quickstart

Apple Silicon Mac + [uv](https://docs.astral.sh/uv/). One command — no clone, no
config, no API key, [no model picker](docs/design.md#why-theres-no-model-picker):

```bash
uvx chad-code     # runs chad anywhere — the command is still `chad`
```

Then make him prove it — a two-minute smoke test on your own machine, offline:

```bash
uvx chad-code prove    # 4 tiny fix-it tasks, mechanically verified, timed 🗿
```

First run only: chad picks the right Ornith model for your RAM (9B under 32 GB,
35B at 32 GB+), asks, and downloads it once into the shared Hugging Face cache —
**~5 GB / ~12 GB: minutes on fast fiber, ~20 on a 100 Mbit line, resumable**.
While it downloads, `cd` into a project you want chad working on and think of a
scoped first ask — *"fix the failing test in `tests/test_x.py`"* lands; *"improve
my codebase"* flails. (The PyPI distribution is **`chad-code`**; bare `chad` is a
squatted, unrelated package.)

Every other way to install — keep it on your PATH, bleeding-edge `main`, the dev
clone — plus the model table and upgrade commands live in
[Installing & upgrading](#installing--upgrading).

chad has some of the same moves as Claude — tool use, plan mode, a real TUI — but driven
by a local model on your laptop instead of a frontier model in a datacenter. He isn't a
smaller Claude; he's a blunter instrument.  

|                     | **Claude**                                       | **chad** 🗿                                |
|---------------------|--------------------------------------------------|--------------------------------------------|
| **Range**           | every workflow, every person, incredible nuance  | one job: code, on your machine             |
| **Runs**            | anywhere — cloud, IDE, terminal, phone           | your laptop. that's it.                    |
| **Brain**           | a frontier model in a datacenter                 | Ornith model on Apple Silicon              |
| **Disposition**     | understands what you *really* meant              | does what you *said*                       |
| **Harness**         | open-ended, anything you can imagine             | plan. execute. nothing else.               |
| **When wrong**      | reasons a way out                                | already shipped 🗿                         |

Think of it as report cards. The exam is
[Terminal-Bench](https://www.tbench.ai/leaderboard), the standard benchmark for CLI coding
agents. chad won't top it — the A students are frontier models in datacenters, and chad
isn't trying to be one. The number worth looking at isn't chad's rank; it's **how much
capability Ornith wrings out of 35B parameters**. On Terminal-Bench 2.0, chad + Ornith
lands right at the **Claude Sonnet 4.5** line — matching open models many times its size,
and beating every open model in its own weight class by a wide margin. **Sonnet on your
laptop.** That's the chart up top.

And you don't have to take our word for it: the whole benchmark is **publicly
reproducible, from a Mac**. The exact Harbor agent adapter that produces our runs, the
runner script, and the step-by-step recipe (serve Ornith on the Mac with
`mlx_lm.server`, point the kit at it) live in
[`benchmarks/tb2/`](benchmarks/tb2/README.md).

> Placeholder numbers — the full verified run is **still in flight**, so treat the Ornith
> point as provisional until it lands (the reproduction kit is published ahead of the
> final number on purpose: check it, don't trust it). The claim the chart makes is
> structural, not one data point: on a
> laptop, *capability per parameter* is the axis you actually compete on, and chad + Ornith
> is out on that frontier by itself. The bet was never that chad out-scores Claude — it's
> that this much capability, running free on your own machine, is worth having around. 🗿

> chad doesn't run on a H100 server, and you don't have one.  He will never will be in the 
> big leagues.  Those are for Claude. chad does **one** thing, MLX inference on a MacBook Pro, 
> and this whole repo is about making that one thing as fast as the laptop allows: a persistent
> prefix KV cache so prefill never re-reads the transcript, and a model that decodes near the
> memory-bandwidth ceiling. chad is not here to win SOTA benchmarks. He's just here to help. 🗿

![chad fixing a failing test end to end — reason, read, edit, rerun pytest, all on a local 35B](docs/demo.gif)

> Real session, unedited: a local 35B (the ≥32 GB tier) reasons through the failure, edits
> the file, reruns the tests, and confirms green. The cold model load is cut; everything
> after is real time.

A single-user agentic coding backend that runs **entirely locally on Apple Silicon**
via [MLX](https://github.com/ml-explore/mlx). It gives you Claude-Code-style tool use
(bash, read, write, edit, glob, grep) driven by a local model — and, increasingly, a
Claude-Code-style **feel**: a full-screen TUI with shift-tab plan mode, a type-ahead
message queue, mid-turn interrupt, and live throughput/context status.

It targets the machine most developers actually have — a **24 GB Apple Silicon
MacBook** (where it runs the 9B; the 35B is the 32 GB+ tier) — and the whole design
follows from one constraint that machine imposes: **prefill is the dominant cost** in an
agentic loop, so a persistent prefix KV cache that never re-reads the transcript is the
core engineering. The full story is in [Design & internals](docs/design.md).

## The bet: at this end of the report card, the harness beats the model

Every serious coding harness was designed for a frontier model behind a datacenter
API. That design bakes in two assumptions: the model is an A student, and prefill is
somebody else's electricity. Both are false on a laptop. A C+ student emits tool calls
with typos, quotes edits it never applies, and rambles — and every token of transcript
it drags around must be re-read by *your* GPU at a few hundred tokens a second.

So chad's real thesis isn't "run a model locally" — plenty of tools do that. It's that
**for a small model, harness quality is worth more than a model upgrade**, and that the
harness and the inference engine have to be designed *together*. Point the same local
weights at a harness built for a frontier API — reaching the model through an OpenAI-style
local server, the standard way to attach one to a local backend — and it stumbles, not
because it's a bad harness but because it was built for an A student on the far end of a
fast API, not a C+ student sharing your GPU.

The gap is entirely nameable failure modes. The model pours its edit into the
reasoning channel and the harness drops it. The model asks for a `read` tool the
harness doesn't ship. The harness rejects nested tool arguments a weak model loves to
emit. The context balloons, cache reuse hits 0%, decode falls to 2 tok/s. chad handles
each of these *inside* the harness: tool calls are parsed in four dialects and
repaired, arguments are schema-coerced with an annotated self-repair loop, edits go
through a forgiveness cascade before failing, loop/thrash/verify guards keep the turn
honest — and, above all, the transcript is engineered to remain a **strict token-prefix
of the live KV cache** across every step, because on a local model prefix stability is
a *harness* property, not a server feature. That co-design is the whole moat: it's why
the same C+ student gets real work done under chad and stalls under harnesses built for
the A student. A C+ student with a good tutor, running free on your own machine. 🗿

### Why the engine is in-process (and not behind an OpenAI layer)

A fair question is whether chad should talk to its model through an OpenAI-style
`/v1/chat/completions` boundary — the way every other local-model harness does — so the
frontend and backend decouple cleanly. It shouldn't, and the co-design is why:
that boundary is stateless text-in/text-out, and chad's core engineering (diffing
rendered *token ids* against a live KV cache, warm-prefix disk checkpoints,
`close_unclosed_think` prefix repair, interruptible chunked prefill, RAM-aware context
sizing, cache push/pop) all require *owning* the tokenizer and the cache object. None of
it survives the API. The coupling isn't debt; it's the measured moat.

To keep that answer honest rather than asserted, chad ships a thin, flag-gated adapter
(`--backend openai`, `src/chad/openai_engine.py`) that runs the *same harness* against
any OpenAI-compatible endpoint — so "harness value" and "engine value" can be measured
separately as an ablation arm. The adapter's honest degradations (it can't report cache
reuse, can't show prefill progress, and interrupts by dropping the stream) are documented
in-code. A measured in-process vs. served comparison (the same harness through
`mlx_lm.server`) is queued on the maintainer's eval rig; the expected result is a
pass-rate near-equal (the harness carries it), with wall-clock/TTFT materially worse.

## Installing & upgrading

The one-line quickstart is up top: `uvx chad-code`. The other ways in, when you
want them:

Install it for good — then it's just `chad`:

```bash
uv tool install chad-code   # install to ~/.local/bin
chad                        # then it's just `chad`
```

To run the very latest **unreleased `main`** instead of the last release, use the git URL:

```bash
uvx --from git+https://github.com/nathansutton/chad chad   # bleeding edge, no clone
```

Or, working from a clone (the dev path):

```bash
uv sync                      # install deps + the `chad` entrypoint (one time)
uv run chad                  # launch the full-screen TUI
uv run chad "add a --json flag to main.py and update the tests"   # one-shot, headless
uv run chad -c               # resume this directory's last conversation
```

**Development / testing.** `uv sync` is the one-time setup. The fast unit gate is
`uv run pytest -q` — it loads **no model weights**, runs in seconds, and is what CI runs.
To measure throughput on your own machine, `uv run chad-bench` (see
[Throughput & performance](docs/benchmarks.md)).

**Optional: precise refs/rename.** LSP-precise cross-file find-references and
scope-correct rename need the `lsp` extra — `uv tool install 'chad-code[lsp]'` (installed)
or `uv sync --extra lsp` (from a clone); without it chad uses the tree-sitter fallback
automatically (see [the symbolic stack](docs/design.md)).

**The model.** chad picks one model for you and downloads it once into the shared Hugging
Face cache (`~/.cache/huggingface`, reused across every project):

| Your Mac | Model | Footprint |
|---|---|---|
| **≥ 32 GB** | [Ornith-1.0-35B `UD-Q2_K_XL`](https://huggingface.co/nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX) — 35B MoE, 2-bit experts | ~12 GB resident (~14 GB with KV + runtime) |
| **16 / 18 / 24 GB** (default) | [Ornith-1.0-9B `UD-Q4_K_XL`](https://huggingface.co/nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX) — 4-bit AWQ | ~5 GB |

The 35B's ~14 GB working set needs headroom the 24 GB machine doesn't have — dogfooding
SIGKILLed it mid-turn there — so the floor for the 35B is **32 GB**; a 24 GB Mac (and the
16/18 GB MacBook Pros) runs the 9B, which fits easily. chad detects your RAM and chooses;
the first run asks before downloading (~12 GB / ~5 GB —
2-4 minutes on gigabit fiber, 15-25 on a 100 Mbit line; the download is resumable, so a
killed first run picks up where it left off), or auto-downloads when headless. No model
picker, no flags. Override with `CHAD_MODEL=<repo
or local dir>` to force a specific one. Quant names follow
[Unsloth's dynamic-quant convention](https://docs.unsloth.ai/) (`UD-…`) so the scheme is
recognizable. chad downloads the pre-quantized model from Hugging Face on first run.

That's the whole on-ramp. The model and the throughput numbers you can reproduce live in
[Throughput & performance](docs/benchmarks.md).

Run `chad` from inside the project you want it to work on — it snapshots the working
directory into context at startup.

### Upgrading

How you refresh depends on how you installed it:

- **`uv tool install` users**: `uv tool upgrade chad-code` re-resolves and installs the
  latest release. (If chad isn't a uv tool, uv tells you so — install it with the `uv tool
  install` line above.)
- **`uvx` users**: `uvx` caches the resolved environment, so a plain re-run can stay
  pinned to an older resolve. Force the latest release with `uvx --refresh chad-code`.
- **Bleeding-edge (`main`)**: to jump ahead of the last release, re-run the git-URL form
  with `--refresh`: `uvx --refresh --from git+https://github.com/nathansutton/chad chad`.
- **Dev clones**: `git pull && uv sync`.

What changed lands in [`CHANGELOG.md`](CHANGELOG.md). Model weights are versioned
separately from the code: a code upgrade never re-downloads the model, and a *model* bump
announces itself in the changelog (superseded snapshots can then be freed — see
[Troubleshooting](docs/troubleshooting.md)).

## Interactive UX (Claude Code parity)

`uv run chad` launches a full-screen terminal UI (`tui.py`, built on prompt_toolkit):

- **shift-tab cycles permission modes** — `normal` (confirm each bash/write/edit) →
  `auto-accept edits` → `plan mode` (read-only: the agent investigates and proposes a
  numbered plan, all mutations blocked) → back. The current mode shows in the status bar.
- **type-ahead message queue** — keep typing while the agent works; messages run in
  order. The status bar shows the queued count.
- **ctrl-c interrupts the running turn** (stops generation) without killing the session.
- **inline y/n approval** for mutating tools in normal mode.
- **live status line** — model, mode, context %, and a live activity readout: a state
  glyph + verb (Reading / Editing / Running…), elapsed seconds, and **↑prefilled /
  ↓generated** token counts. When the prefix cache can't be reused and a full prefill is
  unavoidable (after a `/compact` or a truncation invalidates the cached prefix), it shows
  an advancing **%** so that otherwise-silent re-prefill is legible. (Raw tok/s and
  PLD-acceptance diagnostics still go to `~/.chad/session.log`.)
- **slash commands** — `/init` (scaffold a `CLAUDE.md` from the actual project files),
  `/skills` (list discovered Agent Skills), `/mcp` (list configured MCP servers + their
  tools; `/mcp trust` trusts this project's `./.mcp.json` servers so they start, `/mcp
  login <server>` does an OAuth login), `/accept` (accept a pending plan and implement it),
  `/resume` (list this directory's recent sessions; `/resume <n>` forks one — see
  [Sessions](#sessions-resume--fork)), `/reset` (`/clear`), `/compact` (reclaim context
  now — strips old reasoning + truncates old tool outputs, never drops a message),
  `/model` (model + context status), `/mode`, `/help`, `/exit`. Same set in the `--repl`
  line interface.
- **`@file` / `@dir` mentions** — write `@path` in a message and a file is pulled into
  context inline (bounded by the read tool's skeleton/cap policy, no `read` round-trip) or
  a directory becomes a short listing. Works in the TUI, `--repl`, and `-p`. Emails/
  decorators don't trigger it (the `@` must follow whitespace); only real paths attach.
- **`!command` shell passthrough** — a line starting with `!` runs that shell command
  directly (interruptible) and shows the output, without invoking the model — for quick
  `!git status` / `!ls` checks without leaving the prompt. (TUI and `--repl`.)

The agent loop runs on a worker thread; the UI owns the asyncio loop. Agent I/O
(`emit`/`confirm`/`should_stop`) is injected, so the *same* `Agent` code drives both the
TUI and the plain line REPL (`--repl`). The loop stays fast on a **persistent prefix KV
cache**: each step re-renders the whole transcript but only prefills the handful of new
tokens. The few moments it *can't* — Ornith's hybrid SSM/attention cache is non-trimmable,
so `/compact` or a mid-turn truncation invalidates the prefix and forces a one-time full
re-prefill — are exactly when the status line's advancing **%** earns its place: the long
prefill is the price of the cache design, now shown rather than silent (see
[Design & internals](docs/design.md)).

## Usage

One entrypoint, a handful of flags. `uv run chad --help` is the source of truth:

```bash
uv run chad                  # full-screen TUI (shift-tab for modes, type to queue, ctrl-c to interrupt)
uv run chad "do the thing"   # one-shot headless task, then exit
uv run chad prove            # two-minute offline smoke test: 4 tasks, verified, timed
uv run chad -c               # resume this directory's most recent conversation
uv run chad -c "now also add the --verbose flag"   # resume and continue headless
uv run chad --resume         # list this directory's recent sessions, pick one by number
echo "fix the typo in greet.py" | uv run chad "$(cat)"   # pipe a task in
```

| Flag | What it does |
|---|---|
| `-c, --continue` | resume this directory's **most recent** session (non-destructive — see [Sessions](#sessions-resume--fork)) |
| `--resume` | list the directory's recent sessions and pick one by number (interactive TTY only) |
| `--plan` | start in read-only plan mode (investigate + propose, all edits blocked) |
| `--yolo` | auto-approve bash/write/edit (skip confirm prompts) |
| `--no-think` | skip Ornith's `<think>` blocks — faster on well-scoped work (thinking is on by default) |
| `--repl` | plain line REPL instead of the TUI |

Curated on purpose — `uv run chad --help` is the full set. Notable extras it lists:
`--think-budget` / `--turn-budget-tokens` / `--turn-budget-s` / `--auto-continue` (the
think-cap and runaway-turn governor — see the [Configuration reference](docs/configuration.md#turn-budgets--think-cap));
`--backend openai` / `--base-url` / `--api-key-env` (a research spike that runs the
harness against an OpenAI-compatible endpoint); and `--version` (prints `chad 0.1.0`
plus the checkout's commit — quote it in bug reports).

No model flag: chad runs Ornith (the RAM-appropriate size — see
[Installing & upgrading](#installing--upgrading); the reasoning is
[Why there's no model picker](docs/design.md#why-theres-no-model-picker)).
A headless task (positional, or piped with no TTY) auto-approves mutating tools — otherwise
the confirm prompt would EOF and no file could ever change. Use `--plan` for a read-only
investigation.

The model runs greedy (temp 0). On first run chad downloads it from Hugging Face into
`~/.cache/huggingface` (~12 GB for the 35B, ~5 GB for the 9B); thereafter it loads from
that cache.

The rarely-touched tuning knobs (`CHAD_MAX_CONTEXT`, `CHAD_KV_BITS`, `CHAD_MODEL`, the
turn-budget/think-cap and alternate-backend knobs, the safety/A-B opt-outs, and the
session-log controls) live in environment variables, fully documented in the
[Configuration reference](docs/configuration.md).

### Sessions (resume + fork)

Every conversation is persisted as JSON under `~/.chad/sessions/<cwdhash>/<session_id>.json`
(one file per session, `0600`, atomic write — the store holds full tool args/results, so
it is never world-readable). chad keeps **multiple sessions per directory**, so a project
can carry more than one thread of work:

- **`chad -c`** resumes the **most recent** session for this directory — the simple case,
  unchanged. Now non-destructive: older sessions are kept, not overwritten.
- **`chad --resume`** (and **`/resume`** in the TUI) lists the last ~10 sessions —
  `2h ago · 14 turns · "fix the flaky retry test…"` — and you pick one by number. In the
  TUI, `/resume` prints the numbered list and `/resume <n>` selects. `--resume` needs an
  interactive terminal; headless, use `-c`.
- **Implicit fork.** Resuming *any* session mints a **new** session id seeded with the old
  messages (copy-on-resume) and writes to a new file — the original is never rewritten. So
  every resume is a branch: go back to a session from before a bad turn and continue it
  without destroying either thread. There is no separate fork command; that's the whole
  feature.
- **Retention.** The newest 20 sessions per directory are kept; older ones are pruned on
  save. A tiny per-directory `index.json` (title / timestamp / turn count) makes listing
  cheap, and a pre-existing single-slot session file is migrated into the new layout
  automatically the first time you list. Resume still pays a cold re-prefill of the
  restored transcript (only the message list is stored, not the KV cache).

## Extending chad

chad speaks the same two extension formats as Claude Code:

- **[Agent Skills](https://agentskills.io)** — drop a `SKILL.md` folder in
  `./.claude/skills/` (or `~/.claude/skills/`) and chad discovers it, loading the full
  instructions only when a task matches (progressive disclosure keeps context small).
- **[MCP servers](https://modelcontextprotocol.io)** — configure stdio or HTTP servers in
  `./.mcp.json` (or `~/.chad/mcp.json`) to expose external tools (GitHub, Postgres, Linear,
  Slack, …) alongside chad's builtins, with static-token and OAuth auth.

Both are covered in full — discovery rules, config schema, OAuth flow, the harness
behavior — in the [Configuration reference](docs/configuration.md).

## Documentation

- **[Design & internals](docs/design.md)** — why prefill is the bill, the persistent
  prefix cache, the trimmable/append-only cache trade, why a language server matters on a
  small model, architecture, and the ideas borrowed from other agents.
- **[Throughput & performance](docs/benchmarks.md)** — prefill / decode / warm-step numbers
  you can reproduce with `chad-bench`, the bandwidth ceiling, and the thinking-budget / PLD
  levers.
- **[Terminal-Bench 2.0 reproduction](benchmarks/tb2/README.md)** — the exact Harbor
  adapter and runner behind the chart up top; serve Ornith yourself and check the number.
- **[Configuration reference](docs/configuration.md)** — Agent Skills, MCP servers, the
  context window, every environment variable, and the safety opt-outs.
- **[Troubleshooting](docs/troubleshooting.md)** — when a session rambles, loops, or slows:
  the symptom→knob map for a small local model.
- **[Contributing](CONTRIBUTING.md)** — what lands easily, and what needs a conversation
  first (behavior changes are eval-gated on a private rig).
