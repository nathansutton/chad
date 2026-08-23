# chad — a Claude-Code-style coding agent that fits on a 24 GB MacBook

[![tests](https://github.com/nathansutton/chad/actions/workflows/tests.yml/badge.svg)](https://github.com/nathansutton/chad/actions/workflows/tests.yml)

> Claude can do anything, for anyone, anywhere. chad does one thing. 🗿
> *Coding under supervision.*

A single-user coding agent that runs **entirely on a 24 GB Apple Silicon laptop** via
[MLX](https://github.com/ml-explore/mlx). Plan mode, permission modes, a full-screen TUI,
`@file` mentions, skills, MCP — the Claude Code workflow — driven by one good open model on
your SSD. No Docker, no API key, no model picker, nothing leaves the machine. (Not
affiliated with Anthropic; the shape is borrowed, the code is not.)

## Quickstart

Apple Silicon Mac + [uv](https://docs.astral.sh/uv/). One command — no clone, no config:

```bash
uvx chad-code          # runs chad anywhere — the command is still `chad`
uvx chad-code prove    # offline smoke test: 4 tiny fix-it tasks, verified, timed 🗿
```

First run asks, then downloads the model once (~13 GB, resumable) into the shared Hugging
Face cache. While it downloads, `cd` into a project and think of a scoped first ask —
*"fix the failing test in `tests/test_x.py`"* lands; *"improve my codebase"* flails.

> **chad targets 24 GB Apple Silicon and nothing smaller.** It runs below that and will
> tell you it is doing so, but the model needs ~13 GB resident before a single token of
> context, so a 16 GB Mac gets a window too small to work in.

> The PyPI package is **`chad-code`**; bare `chad` is an unrelated squatted package. Other
> ways in (PATH install, bleeding-edge `main`, dev clone) are in
> [Installing & upgrading](#installing--upgrading).

## What chad is

The open models got good. [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) writes
correct code, uses a shell properly, and fits — quantized — on a laptop. Nobody needs
convincing of that anymore. What a laptop still lacks is a harness that makes that model
**feel** like the tool you already use. chad is that harness, and it is built around two
commitments:

1. **A Claude-Code-shaped developer experience for one scoped task at a time.** The same
   TUI moves — shift-tab through `normal` / `auto-accept` / `yolo` / `plan`, an approval
   prompt that shows the whole command, `@file`, `!cmd`, `/undo`, Agent Skills, MCP — so
   nothing has to be relearned. One model, one command, zero decisions before the first ask.
2. **A 24 GB MacBook Pro is the whole machine.** ~13 GB of weights plus a ~4 GB prefill
   transient leaves no slack, so the context window is sized from *live* memory, and every
   token of it is spent on purpose: a ~2.8k-token system prompt, tool output clipped with a path back to
   the full body, a todo list that is a markdown checklist, no skill catalog in the prompt.

Everything that serves those two commitments got the engineering budget; everything else is
deliberately plain.

**Where the budget went — tokens per second.** A dense 27B at 3-bit streams ~12 GB of
weights per token, which on an M4 Pro is ~18 tok/s serially, and an agent that decodes at
18 tok/s while re-reading its transcript every step is a batch job, not a pair programmer.
chad owns its inference loop instead of talking to a server, and the speed comes from three
things fitted to the shipped checkpoint:

- **DFlash2 block speculation** — a bundled 1.9B drafter ([z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
  ported to MLX and quantized) proposes a whole block of tokens from the main model's own
  hidden states; the main model verifies the block in one batched forward; exact rejection
  sampling keeps every emitted token the model's own. Decode goes ~3× on prose and code,
  ~2× on real mid-session agent contexts.
- **A persistent prefix KV cache** — the transcript is kept a strict token-prefix of the
  live cache, so a follow-up step prefills the ~16 tokens it appended, not the 5,000 it
  already read: **~0.75 s per step instead of ~50 s**. Any server with prompt caching gets
  the easy case; the work is keeping it true across compaction, truncated turns and
  restarts — the stable system prefix is checkpointed to disk, so the second session in a
  project starts warm (75.6 s → 5.5 s to the first tool call).
- **Fused Metal kernels** — quantized-KV attention, a small-M matmul for speculative
  verify, a compiled single-token layer step — chosen per machine at load time, no knobs.

**Where it didn't — everything else is KISS.** Five tools: `bash`, `edit`, `write`,
`write_todos`, `done`. `bash` is the primary tool because the model already knows `rg`,
`sed -n`, and your test runner from pretraining, and a dedicated `read`/`grep`/`glob` family
measured no better than the shell it replaced. The todo list is a checklist the model
copies forward and ticks. There are no behavioral levers to tune. The whole harness is
readable in an afternoon.

**What it isn't.** A 3-bit 27B is not a frontier model: it lands a scoped ask and flails on
a vague one. The usable window on 24 GB is tens of thousands of tokens, not the 262k on the
label. The first turn in a *new* project spends over a minute prefilling before the warm
start exists. Thinking is on by default and is most of what it generates. It needs a 24 GB
Apple Silicon Mac, full stop. The docs quote no benchmark scores — only throughput you can
reproduce on that Mac — because that is the only machine the claim is about.

## chad is not a baby Claude

He has some of the same moves — tool use, plan mode, a real TUI — but he's a blunter
instrument:

|                 | **Claude**                                      | **chad** 🗿                     |
|-----------------|-------------------------------------------------|---------------------------------|
| **Range**       | every workflow, every person, incredible nuance | one job: code, on your machine  |
| **Runs**        | anywhere — cloud, IDE, terminal, phone          | your mac. that's it.            |
| **Brain**       | a frontier model in a datacenter                | one 27B on your SSD             |
| **Disposition** | understands what you *meant*                    | does what you *said*            |
| **Harness**     | open-ended, anything you can imagine            | plan. execute. nothing else.    |
| **When wrong**  | reasons a way out                               | already shipped                 |

![chad fixing a failing test end to end — reason, read, edit, run pytest, confirm green, all on a local model](docs/demo.gif)

> Real session, unedited (the silent prefill is cut): a local model finds the cent that
> floor division loses, fixes it, and verifies itself — then your own `pytest`, in your
> own shell. Recorded with `--yolo` so nothing pauses for a keypress; the default mode
> stops and asks before every edit and every command.

## Same model, same Mac, stock engine

The fair question is not "how does chad compare to Claude" — it doesn't — but "what do I
gain over pointing a generic local-model tool at the same weights?" So: Qwen3.8-27B at the
same `UD-Q3_K_XL` recipe (Unsloth's GGUF for llama.cpp, chad's MLX conversion of the same
bit map), the same M4 Pro (24 GB), one engine resident at a time, each measured with its own
benchmark on a 512-token prompt and a 128-token generation. Ollama is not a separate row:
it is llama.cpp underneath, with no speculative decoding for this model — measured once on
the same GGUF (0.32.15, Modelfile `FROM` only): 96 tok/s prefill, **10.9** decode, the
same number.

| Engine | Prefill (512-tok prompt) | Decode (128 tok) | Speculative decoding |
|---|---|---|---|
| llama.cpp `llama-bench` (stock, build 10470) | 102 tok/s | 10.9 tok/s | none for this model |
| **chad**, serial (`CHAD_NO_DFLASH=1`) | 99 tok/s | 18.1 tok/s | off |
| **chad**, default | 98 tok/s | **62 tok/s**¹ | DFlash2 block drafter |

¹ `chad-bench`'s prompt is tiled code the drafter reads easily, so 62 is a ceiling. On ten
real mid-session contexts replayed from `~/.chad/sessions` (12–19k tokens, tool results and
schemas in place, 384-token decodes) the same engine measures **31.7 tok/s median / 21.4
floor** greedy against 14.8 serial, and **27.6 / 17.7** at the default thinking preset
against 13.9 — much of a real turn is `<think>`, where the drafter lands 35–55% of its
proposals. That ~2× is the number a session lives at. The serial gap to llama.cpp is
partly bytes (the MLX quant is ~12 GB against the GGUF's 13.1) and partly chad's fused
single-token kernels; llama.cpp's Metal path for this hybrid architecture was not profiled,
so read that row as what the fitted engine buys, not as a verdict on llama.cpp.

Method, the longer runs and the caveats are in
[Throughput & performance](docs/benchmarks.md#same-model-same-mac-stock-engine); the rows
are committed under [`benchmarks/stock/_runs/`](benchmarks/stock/); reproduce them with
`uv run python benchmarks/stock/stock.py {llama,chad}`.

## Installing & upgrading

The one-line quickstart (`uvx chad-code`) is up top. The other ways in:

```bash
uv tool install chad-code   # install for good — then it's just `chad`
uvx --from git+https://github.com/nathansutton/chad chad   # bleeding-edge main, no clone
```

Or from a clone (the dev path):

```bash
uv sync                      # install deps + the `chad` entrypoint (one time)
uv run chad                  # full-screen TUI
uv run chad "add a --json flag to main.py and update the tests"   # one-shot, headless
uv run chad -c               # resume this directory's last conversation
```

**Optional extras.** Everything core is in the base install; two features are opt-in
because they pull deps not every install wants — `speech` (voice mode: a mic library,
no torch) and `highlight` (syntax colour in diffs/previews). An extra rides on the
**install spec**, not on a separate command, so how you add it depends on how you
installed chad:

```bash
uv tool install --force 'chad-code[speech]'   # add to an existing `uv tool` install
uvx --from 'chad-code[speech]' chad           # one-off run, nothing installed
uv sync --extra speech                        # from a clone
```

`/speech` in the TUI prints whichever of those matches your install, so you never have
to work it out from here.

**The model.** chad ships exactly one, downloaded once into the shared Hugging Face cache
(`~/.cache/huggingface`, reused across every project). There is no picker and no size tier.

| Model | Quant | Footprint |
|---|---|---|
| [Qwen3.8-27B `UD-Q3_K_XL-DFlash2`](https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX) | 3-bit group-64 body, 5-bit `lm_head`, bundled 4-bit DFlash2 drafter | ~13 GB resident, 262k native context |

Qwen3.8-27B is **dense** (64 layers: 48 GatedDeltaNet + 16 full attention), so every
parameter is on the critical path for every token and the quant is where decode speed comes
from. The bits go where measurement says they pay: `lm_head` is a second full 1.27B-param
tensor and is held at 5-bit; `embed_tokens`, a lookup table whose error never compounds
through a matmul, is cheapest. The name follows
[Unsloth's dynamic-quant convention](https://docs.unsloth.ai/) (`UD-…`); the quant itself is
MLX group-64 affine, not a llama.cpp k-quant. The DFlash2 drafter ships inside the same repo,
pre-quantized — one download, and speculative decoding is simply there.

**24 GB is the floor.** The context window is sized from the live Metal budget, not a
constant, so a tighter box narrows its window rather than dying — but ~13 GB of weights
plus a ~4.3 GB prefill transient spend most of a 24 GB budget before the first cached
token, which is why a 24 GB Mac lands in the tens of thousands of tokens rather than the
model's 262k. The banner states what you actually got. Below 24 GB chad warns and proceeds;
it does not gate you, but it cannot give you a usable window either.

`--model <repo or local dir>` runs different weights through the same engine, and stays a
first-class escape hatch: shipping one model is a default, not a cage. What you give up is
fit, not function — the block drafter, the fused-attention coverage, the decode fastpath
and the context governor are all fitted to the shipped checkpoint, so other weights are
slower rather than broken. The flag takes a repo id or a directory; there are no size
shorthands.

**Upgrading** — depends on how you installed: `uv tool upgrade chad-code`; `uvx --refresh
chad-code`; or `git pull && uv sync` for a clone. What changed lands in
[`CHANGELOG.md`](CHANGELOG.md). Model weights are versioned separately — a code upgrade never
re-downloads the model.

**Development.** `uv sync` once, then `uv run pytest -q` — the fast unit gate loads **no
model weights**, runs in seconds, and is what CI runs. Throughput on your own machine:
`uv run chad-bench` (see [Throughput & performance](docs/benchmarks.md)).

## Interactive UX

`uv run chad` launches a full-screen terminal UI (built on prompt_toolkit):

- **shift-tab cycles permission modes** — `normal` (confirm each bash/write/edit) →
  `auto-accept edits` (edits land silently; **terminal commands still ask**) → `yolo`
  (nothing asks) → `plan mode` (read-only: investigate + propose a numbered plan) → back.
- **the approval prompt shows what you're approving** — the full command on its own lines
  above the input, not a clipped single line. Approving blind isn't approving.
- **type-ahead message queue** — keep typing while the agent works; messages run in order.
- **ctrl-c interrupts the running turn** without killing the session.
- **live status line** — model, mode, context %, a state glyph + verb, elapsed seconds, and
  **↑prefilled / ↓generated** token counts (with an advancing **%** on an unavoidable full
  re-prefill, so it's never silent).
- **slash commands** — `/init`, `/skills`, `/<skill>`, `/mcp`, `/accept`, `/resume`,
  `/compact`, `/ctx` (where the window is going, split into system prompt, tool schemas,
  think residue and tool results), `/model`, `/mode`, `/help`, `/exit`, and the two that
  undo the agent:
  **`/undo`** reverts your files to the last edit checkpoint and **`/restore`** lists the
  checkpoints so you can revert to any of them. Same set in the `--repl` line interface.
- **`@file` / `@dir` mentions** and **`!command` shell passthrough** — pull a file into
  context inline, or run a shell command without invoking the model.
- **voice mode, all local** — `/speech`, then ctrl-t to talk: Parakeet-on-MLX transcribes
  into the input box for you to review before Enter sends it (esc discards a take);
  replies are read aloud via macOS `say`. A pre-roll buffer means your first syllable
  isn't clipped, and a personal word table (`~/.chad/speech_words.json`) teaches it your
  identifiers — `{"pie test": "pytest"}`. Dictation cost is linear in take length, so a
  long thought is fine; `/speech` off releases both the mic and the weights. Nothing
  leaves the machine. Needs the `speech` extra — just a mic library; no torch, no numba
  (see [Installing & upgrading](#installing--upgrading); on a `uv tool` install that's
  `uv tool install --force 'chad-code[speech]'`).

**Usage.** `uv run chad --help` is the source of truth:

| Flag | What it does |
|---|---|
| `-c, --continue` | resume this directory's **most recent** session (non-destructive) |
| `--resume` | list recent sessions, pick one by number (interactive TTY only) |
| `--plan` | start in read-only plan mode (investigate + propose, edits blocked) |
| `--yolo` | auto-approve bash/write/edit (skip confirm prompts) |
| `--no-think` | skip the model's `<think>` blocks — faster on well-scoped work |
| `--model` | `auto` (the shipped default), or any HF repo id / local model dir |
| `--repl` | plain line REPL instead of the TUI |

Plus three subcommands, each with its own `--help`:

| Command | What it does |
|---|---|
| `chad serve` | serve this Mac's model to a container or the LAN ([Configuration](docs/configuration.md#serving-the-local-model-to-a-container-chad-serve)) |
| `chad prove` | offline smoke test: 4 tiny fix-it tasks, verified, timed |
| `chad levers` | print the result-channel lever registry as JSON (for A/B ablation) |

A headless task (positional, or piped with no TTY) auto-approves mutating tools; the model
runs greedy (temp 0). Every conversation is persisted under `~/.chad/sessions/`, and every
resume forks a new branch rather than overwriting — details in
[Configuration](docs/configuration.md#sessions). The rarely-touched tuning knobs
(`CHAD_MAX_CONTEXT`, `CHAD_KV_BITS`, turn-budget/think-cap, safety opt-outs) all live in
environment variables, fully documented there.

## Extending chad

chad speaks the same two extension formats as Claude Code:

- **[Agent Skills](https://agentskills.io)** — drop a `SKILL.md` folder in `./.claude/skills/`
  and it becomes a slash command: `/ship`, `/investigate the flaky test`. Skills cost
  nothing until you run one — chad puts no skill catalog in the system prompt.
- **[MCP servers](https://modelcontextprotocol.io)** — configure stdio or HTTP servers in
  `./.mcp.json` to expose external tools (GitHub, Postgres, Linear, Slack, …) alongside
  chad's builtins, with static-token and OAuth auth.

Both are covered in full in the [Configuration reference](docs/configuration.md).

## Documentation

- **[Design & internals](docs/design.md)** — why prefill is the bill, the persistent prefix
  cache, the trimmable/append-only trade, why the tool surface is five tools, and the ideas
  borrowed from other agents.
- **[Throughput & performance](docs/benchmarks.md)** — prefill / decode / warm-step
  numbers you can reproduce with `chad-bench`, the stock-engine comparison, and what the
  cross-session warm start is worth.
- **[Configuration reference](docs/configuration.md)** — Agent Skills, MCP servers, the
  context window, every environment variable, and the safety opt-outs.
- **[Troubleshooting](docs/troubleshooting.md)** — when a session rambles, loops, or slows:
  the symptom→knob map for a small local model.
- **[Contributing](CONTRIBUTING.md)** — what lands easily, and what needs a conversation first.
