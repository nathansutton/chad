# chad: a Claude-Code-style coding agent on your lap

[![tests](https://github.com/nathansutton/chad/actions/workflows/tests.yml/badge.svg)](https://github.com/nathansutton/chad/actions/workflows/tests.yml)

<img src="docs/claude-vs-chad.jpg" width="66%" alt="Two staircase newel posts side by side: Claude is a hand-carved wooden horse head, chad is a scuffed plastic toy horse tied on with twine">

> Claude can do anything, for anyone, anywhere. chad does one thing. 🗿
> *Coding under supervision.*

chad is a single-user coding agent that runs entirely on an Apple Silicon Mac via
[MLX](https://github.com/ml-explore/mlx). One 27B model and no API key. Every other 
harness assumes a datacenter on the other end of a socket; chad assumes a laptop, 
and the whole design falls out of that. (Not affiliated with Anthropic.)

## Try it

```bash
uvx chad-code          # runs anywhere; the command is still `chad`
uvx chad-code prove    # offline smoke test: 4 tiny fix-it tasks, verified, timed 🗿
```

The first run asks, then downloads the model once (~13 GB) into the shared Hugging
Face cache. While it downloads, `cd` into a project and think of a scoped first ask:
*"fix the failing test in `tests/test_x.py`"* lands, *"improve my codebase"* flails.

chad targets 24 GB and nothing smaller. It runs below that and tells you it is doing so,
but 13 GB of weights sit resident before a single token of context, so a 16 GB Mac gets a
window too small to work in.

The PyPI package is `chad-code`. Bare `chad` is an unrelated squatted package.

![chad fixing a failing test end to end: reason, read, edit, run pytest, confirm green, all on a local model](docs/demo.gif)

> Real session, unedited (the silent prefill is cut). A local model finds the cent that
> floor division loses, fixes it, and verifies itself, then runs your own `pytest` in your
> own shell. Recorded with `--yolo` so nothing pauses for a keypress; the default mode stops
> and asks before every edit and every command.

## Same model, same Mac, stock engine

The question worth answering: what do you gain over pointing a generic local-model tool at
the same weights? Qwen3.8-27B at the same `UD-Q3_K_XL` recipe (Unsloth's GGUF for
llama.cpp, chad's MLX conversion of the same bit map), the same M4 Pro (24 GB), one engine
resident at a time, each measured with its own benchmark on a 512-token prompt and a
128-token generation.

| Engine | Prefill (512-tok prompt) | Decode (128 tok) | Speculative decoding |
|---|---|---|---|
| llama.cpp `llama-bench` (stock, build 10470) | 102 tok/s | 10.9 tok/s | none for this model |
| **chad**, serial (`CHAD_NO_DFLASH=1`) | 99 tok/s | 18.1 tok/s | off |
| **chad**, default | 98 tok/s | **62 tok/s**¹ | DFlash2 block drafter |

A 200-token function body takes roughly 18 seconds at 10.9 tok/s and 3 at 62. That is the
difference between a batch job and a pair programmer, and closing it is what the project is
for.

Ollama does not get its own row: it is llama.cpp underneath with no speculative decoding for
this model, and on the same GGUF (0.32.15, Modelfile `FROM` only) it measures 96 tok/s
prefill and the same **10.9** decode.

¹ 62 is a ceiling: `chad-bench`'s prompt is tiled code the drafter reads easily. Replayed
against ten real mid-session contexts from `~/.chad/sessions` (12–19k tokens, tool results in
place, 384-token decodes) the same engine measures **31.7 tok/s median / 21.4 floor** greedy
against 14.8 serial, and **27.6 / 17.7** thinking against 13.9. That ~2× is what a session
lives at. llama.cpp's Metal path for this hybrid architecture was not profiled, so read its
row as what a fitted engine buys, not as a verdict on llama.cpp.

Method, the longer runs and the caveats are in
[Throughput & performance](docs/benchmarks.md#same-model-same-mac-stock-engine); the rows are
committed under [`benchmarks/stock/_runs/`](benchmarks/stock/); reproduce them with
`uv run python benchmarks/stock/stock.py {llama,chad}`.

## Where the speed comes from

chad owns its inference loop instead of talking to a server, and the engine is fitted to the
one checkpoint it ships:

- **DFlash2 block speculation.** A bundled 1.9B drafter ([z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2),
  ported to MLX and quantized) proposes a whole block of tokens from the main model's own
  hidden states. The main model verifies the block in one batched forward, and exact
  rejection sampling keeps every emitted token the model's own.
- **A persistent prefix KV cache.** The transcript is kept a strict token-prefix of the live
  cache, so a follow-up step prefills the ~16 tokens it appended instead of the 5,000 it
  already read: **~0.75 s per step instead of ~50 s**. Any server with prompt caching gets
  the easy case; the work is holding it true across compaction, truncated turns and restarts.
  The system prefix is checkpointed to disk, so the second session in a project starts warm
  (75.6 s → 5.5 s to the first tool call).
- **Fused Metal kernels.** Quantized-KV attention, a small-M matmul for speculative verify,
  and a compiled single-token layer step, chosen per machine at load time, no knobs.

## What chad gives up

He has some of the same moves (tool use, plan mode, a real TUI) but he is a blunter
instrument:

|                 | **Claude**                                      | **chad** 🗿                     |
|-----------------|-------------------------------------------------|---------------------------------|
| **Range**       | every workflow, every person, incredible nuance | one job: code, on your machine  |
| **Runs**        | anywhere: cloud, IDE, terminal, phone           | your mac. that's it.            |
| **Brain**       | a frontier model in a datacenter                | one 27B on your SSD             |
| **Disposition** | understands what you *meant*                    | does what you *said*            |
| **Harness**     | open-ended, anything you can imagine            | plan. execute. nothing else.    |
| **When wrong**  | reasons a way out                               | already shipped                 |

Five tools: `bash`, `edit`, `write`, `write_todos`, `done`. `bash` is the primary one
because the model already knows `rg` and `sed -n`. The todo list is a checklist the model
copies forward and ticks. Speed and the Claude Code muscle memory got the engineering
budget; everything else is deliberately plain.

## Interactive UX

`uv run chad` launches a full-screen terminal UI (built on prompt_toolkit):

- **shift-tab cycles permission modes**: `normal` (confirm each bash/write/edit) →
  `auto-accept edits` (edits land silently, **terminal commands still ask**) → `yolo`
  (nothing asks) → `plan mode` (read-only: investigate and propose a numbered plan) → back.
- **type-ahead message queue.** Keep typing while the agent works; messages run in order.
- **ctrl-c interrupts the running turn** without killing the session. **↑prefilled /
  ↓generated** token counts show an advancing **%** on an unavoidable full re-prefill, so it
  is never silent.
- **`@file` / `@dir` mentions** and **`!command` shell passthrough.** Pull a file into
  context inline, or run a shell command without invoking the model.
- **voice mode, all local.** `/speech`, then ctrl-t to talk: Parakeet-on-MLX transcribes into
  the input box for you to review before Enter sends it, and replies are read aloud via macOS
  `say`. A word table teaches it your identifiers. Needs the `speech` extra
  ([details](docs/configuration.md#voice-mode-speech)).

`uv run chad --help` is the source of truth:

| Flag | What it does |
|---|---|
| `-c, --continue` | resume this directory's **most recent** session (non-destructive) |
| `--resume` | list recent sessions, pick one by number (interactive TTY only) |
| `--plan` | start in read-only plan mode (investigate and propose, edits blocked) |
| `--yolo` | auto-approve bash/write/edit (skip confirm prompts) |
| `--no-think` | skip the model's `<think>` blocks, faster on well-scoped work |
| `--model` | `auto` (the shipped default), or any HF repo id / local model dir |
| `--repl` | plain line REPL instead of the TUI |

Three subcommands, each with its own `--help`: `chad prove` (the offline smoke test),
`chad serve` ([serve this Mac's model to a container or the LAN](docs/configuration.md#serving-the-local-model-to-a-container-chad-serve)),
and `chad levers` (print the result-channel lever registry as JSON, for A/B ablation).

A headless task (positional, or piped with no TTY) auto-approves mutating tools and runs
greedy (temp 0). Every conversation is persisted under `~/.chad/sessions/`, and every resume
forks a new branch rather than overwriting.

## The model

chad ships exactly one, downloaded once into the shared Hugging Face cache
(`~/.cache/huggingface`, reused across every project). There is no picker and no size tier.

| Model | Quant | Footprint |
|---|---|---|
| [Qwen3.8-27B `UD-Q3_K_XL-DFlash2`](https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX) | 3-bit group-64 body, 5-bit `lm_head`, bundled 4-bit DFlash2 drafter | ~13 GB resident, 262k native context |

Qwen3.8-27B is **dense** (64 layers: 48 GatedDeltaNet + 16 full attention), so every
parameter is on the critical path for every token and the quant is where decode speed comes
from. The bits go where held-out perplexity says they pay: `lm_head` is a second full
1.27B-param tensor and is held at 5-bit, while `embed_tokens` is a lookup table whose error
never compounds through a matmul, so it is cheapest. The name follows
[Unsloth's convention](https://docs.unsloth.ai/) (`UD-…`), though the quant is MLX group-64
affine, not a llama.cpp k-quant. The drafter ships in the same repo, pre-quantized.

`--model <repo or local dir>` runs different weights through the same engine and stays a
first-class escape hatch. The drafter, the fused-attention coverage, the decode fastpath and
the context governor are all fitted to the shipped checkpoint, so other weights run slower;
they do not break.

## Installing & upgrading

The one-line quickstart (`uvx chad-code`) is up top. The other ways in:

```bash
uv tool install chad-code   # install for good, then it's just `chad`
uvx --from git+https://github.com/nathansutton/chad chad   # bleeding-edge main, no clone
```

Or from a clone (the dev path):

```bash
uv sync                      # install deps + the `chad` entrypoint (one time)
uv run chad                  # full-screen TUI
uv run chad "add a --json flag to main.py and update the tests"   # one-shot, headless
uv run chad -c               # resume this directory's last conversation
```

**Optional extras.** Two features are opt-in because they pull deps not every install wants:
`speech` (voice mode: a mic library, no torch) and `highlight` (syntax colour in diffs and
previews). An extra rides on the install spec, not on a separate command, so how you add it
depends on how you installed chad:

```bash
uv tool install --force 'chad-code[speech]'   # add to an existing `uv tool` install
uvx --from 'chad-code[speech]' chad           # one-off run, nothing installed
uv sync --extra speech                        # from a clone
```

`/speech` in the TUI prints whichever of those matches your install, so you never have to
work it out from here.

**Upgrading** depends on how you installed: `uv tool upgrade chad-code`, `uvx --refresh
chad-code`, or `git pull && uv sync` for a clone. What changed lands in
[`CHANGELOG.md`](CHANGELOG.md). Model weights are versioned separately, so a code upgrade
never re-downloads the model.

**Development.** `uv sync` once, then `uv run pytest -q`. The fast unit gate loads **no model
weights**, runs in seconds, and is what CI runs. For throughput on your own machine, use
`uv run chad-bench` (see [Throughput & performance](docs/benchmarks.md)).

## Extending chad

chad speaks the same two extension formats as Claude Code:

- **[Agent Skills](https://agentskills.io).** Drop a `SKILL.md` folder in
  `./.claude/skills/` and it becomes a slash command: `/ship`, `/investigate the flaky test`.
  Skills cost nothing until you run one, because chad puts no skill catalog in the system
  prompt.
- **[MCP servers](https://modelcontextprotocol.io).** Configure stdio or HTTP servers in
  `./.mcp.json` to expose external tools (GitHub, Postgres, Linear, Slack, …) alongside
  chad's builtins, with static-token and OAuth auth.

Both are covered in full in the [Configuration reference](docs/configuration.md).

## Documentation

- **[Design & internals](docs/design.md)** covers why prefill is the bill, the persistent
  prefix cache, the trimmable/append-only trade, why the tool surface is five tools, and the
  ideas borrowed from other agents.
- **[Throughput & performance](docs/benchmarks.md)** has the prefill, decode and warm-step
  numbers you can reproduce with `chad-bench`, the stock-engine comparison, and what the
  cross-session warm start is worth.
- **[Configuration reference](docs/configuration.md)** documents Agent Skills, MCP servers,
  the context window, every environment variable, and the safety opt-outs.
- **[Troubleshooting](docs/troubleshooting.md)** is the symptom→knob map for when a session
  rambles, loops, or slows.
- **[Contributing](CONTRIBUTING.md)** says what lands easily and what needs a conversation
  first.

